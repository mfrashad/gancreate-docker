try: # set up path
    import sys
    sys.path.append('ganspace')
    sys.path.append('first-order-model')
    sys.path.append('iPERCore')
    sys.path.append('Wav2Lip')
    print('Paths added')
except Exception as e:
    print(e)
    pass


import os
import torch
import face_alignment
import PIL
from PIL import Image
import numpy as np
from skimage.transform import resize
import cv2
import shutil
import warnings
from models import get_instrumented_model
from decomposition import get_or_compute
from config import Config

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict
from pydantic import BaseModel


import uvicorn

warnings.filterwarnings("ignore")
 
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                      device='cuda')

image_size = 512

def create_bounding_box(target_landmarks, expansion_factor=1):
    target_landmarks = np.array(target_landmarks)
    x_y_min = target_landmarks.reshape(-1, 68, 2).min(axis=1)
    x_y_max = target_landmarks.reshape(-1, 68, 2).max(axis=1)
    expansion_factor = (expansion_factor-1)/2
    bb_expansion_x = (x_y_max[:, 0] - x_y_min[:, 0]) * expansion_factor
    bb_expansion_y = (x_y_max[:, 1] - x_y_min[:, 1]) * expansion_factor
    x_y_min[:, 0] -= bb_expansion_x
    x_y_max[:, 0] += bb_expansion_x
    x_y_min[:, 1] -= bb_expansion_y
    x_y_max[:, 1] += bb_expansion_y
    return np.hstack((x_y_min, x_y_max-x_y_min))

def fix_dims(im):
    if im.ndim == 2:
        im = np.tile(im[..., None], [1, 1, 3])
    return im[...,:3]

def get_crop(im, center_face=True, crop_face=True, expansion_factor=1, landmarks=None):
    im = fix_dims(im)
    if (center_face or crop_face) and not landmarks:
        landmarks = fa.get_landmarks_from_image(im)
    if (center_face or crop_face) and landmarks:
        rects = create_bounding_box(landmarks, expansion_factor=expansion_factor)
        x0,y0,w,h = sorted(rects, key=lambda x: x[2]*x[3])[-1]
        if crop_face:
            s = max(h, w)
            x0 += (w-s)//2
            x1 = x0 + s
            y0 += (h-s)//2
            y1 = y0 + s
        else:
            img_h,img_w = im.shape[:2]
            img_s = min(img_h,img_w)
            x0 = min(max(0, x0+(w-img_s)//2), img_w-img_s)
            x1 = x0 + img_s
            y0 = min(max(0, y0+(h-img_s)//2), img_h-img_s)
            y1 = y0 + img_s            
    else:
        h,w = im.shape[:2]
        s = min(h,w)
        x0 = (w-s)//2
        x1 = x0 + s
        y0 = (h-s)//2
        y1 = y0 + s
    return int(x0),int(x1),int(y0),int(y1)

def pad_crop_resize(im, x0=None, x1=None, y0=None, y1=None, new_h=256, new_w=256):
    im = fix_dims(im)
    h,w = im.shape[:2]
    if x0 is None:
      x0 = 0
    if x1 is None:
      x1 = w
    if y0 is None:
      y0 = 0
    if y1 is None:
      y1 = h
    if x0<0 or x1>w or y0<0 or y1>h:
        im = np.pad(im, pad_width=[(max(-y0,0),max(y1-h,0)),(max(-x0,0),max(x1-w,0)),(0,0)], mode='edge')
    return resize(im[max(y0,0):y1-min(y0,0),max(x0,0):x1-min(x0,0)], (new_h, new_w))

def get_crop_body(im, center_body=True, crop_body=True, expansion_factor=1, rects=None):
    im = fix_dims(im)
    if (center_body or crop_body) and rects is None:
        rects, _ = hog.detectMultiScale(im, winStride=(4, 4),padding=(8,8), scale=expansion_factor)
    if (center_body or crop_body) and rects is not None and len(rects):
        x0,y0,w,h = sorted(rects, key=lambda x: x[2]*x[3])[-1]
        if crop_body:
            x0 += w//2-h//2
            x1 = x0+h
            y1 = y0+h
        else:
            img_h,img_w = im.shape[:2]
            x0 += (w-img_h)//2
            x1 = x0+img_h
            y0 = 0
            y1 = img_h
    else:
        h,w = im.shape[:2]
        x0 = (w-h)//2
        x1 = (w+h)//2
        y0 = 0
        y1 = h
    return int(x0),int(x1),int(y0),int(y1)

def crop_resize(im, size, crop=False):
  if im.shape[:2] == size:
    return im
  if size[0]<im.shape[0] or size[1]<im.shape[1]:
    interp = cv2.INTER_AREA
  else:
    interp = cv2.INTER_CUBIC
  if not crop:
    return np.clip(cv2.resize(im, size[::-1], interpolation=interp),0,1)
  ratio = max(size[0]/im.shape[0], size[1]/im.shape[1])
  im = np.clip(cv2.resize(im, (int(np.ceil(im.shape[1]*ratio)), int(np.ceil(im.shape[0]*ratio))), interpolation=interp),0,1)
  return im[(im.shape[0]-size[0])//2:(im.shape[0]-size[0])//2+size[0], (im.shape[1]-size[1])//2:(im.shape[1]-size[1])//2+size[1]]


#@title Define GANSpace functions
from ipywidgets import fixed

# Taken from https://github.com/alexanderkuk/log-progress
def log_progress(sequence, every=1, size=None, name='Items'):
    from ipywidgets import IntProgress, HTML, VBox
    from IPython.display import display

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    progress.value = index
                    label.value = u'{name}: {index} / {size}'.format(
                        name=name,
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = "{name}: {index}".format(
            name=name,
            index=str(index or '?')
        )

def name_direction(sender):
  if not text.value:
    print('Please name the direction before saving')
    return
    
  if num in named_directions.values():
    target_key = list(named_directions.keys())[list(named_directions.values()).index(num)]
    print(f'Direction already named: {target_key}')
    print(f'Overwriting... ')
    del(named_directions[target_key])
  named_directions[text.value] = [num, start_layer.value, end_layer.value]
  save_direction(random_dir, text.value)
  for item in named_directions:
    print(item, named_directions[item])

def save_direction(direction, filename):
  filename += ".npy"
  np.save(filename, direction, allow_pickle=True, fix_imports=True)
  print(f'Latent direction saved as {filename}')

def mix_w(w1, w2, content, style):
    for i in range(0,5):
        w2[i] = w1[i] * (1 - content) + w2[i] * content

    for i in range(5, 16):
        w2[i] = w1[i] * (1 - style) + w2[i] * style
    
    return w2

def display_sample_pytorch(seed=None, truncation=0.5, directions=None, distances=None, scale=1, start=0, end=14, w=None, disp=True, save=None):
    # blockPrint()
    model.truncation = truncation
    if w is None:
        w = model.sample_latent(1, seed=seed).detach().cpu().numpy()
        w = [w]*model.get_max_latents() # one per layer
    else:
        w = [np.expand_dims(x, 0) for x in w]
    
    if directions != None and distances != None:
        for l in range(start, end):
          for i in range(len(directions)):
            w[l] = w[l] + directions[i] * distances[i] * scale
    
    torch.cuda.empty_cache()
    #save image and display
    out = model.sample_np(w)
    final_im = Image.fromarray((out * 255).astype(np.uint8)).resize((500,500),Image.LANCZOS)
    
    
    if save is not None:
      if disp == False:
        print(save)
      final_im.save(f'out/{seed}_{save:05}.png')
    if disp:
      display(final_im)
    
    return final_im



#@title Load Model for GANSpace
selected_model = 'portrait'#@param ["portrait", "character", "model", "lookbook"]



# Speed up computation
torch.autograd.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True

# Specify model to use
config = Config(
  model='StyleGAN2',
  layer='style',
  output_class=selected_model,
  components=80,
  use_w=True,
  batch_size=5_000, # style layer quite small
)

inst = get_instrumented_model(config.model, config.output_class,
                              config.layer, torch.device('cuda'), use_w=config.use_w)

path_to_components = get_or_compute(config, inst)

model = inst.model

comps = np.load(path_to_components)
lst = comps.files
latent_dirs = []
latent_stdevs = []

load_activations = False

for item in lst:
    if load_activations:
      if item == 'act_comp':
        for i in range(comps[item].shape[0]):
          latent_dirs.append(comps[item][i])
      if item == 'act_stdev':
        for i in range(comps[item].shape[0]):
          latent_stdevs.append(comps[item][i])
    else:
      if item == 'lat_comp':
        for i in range(comps[item].shape[0]):
          latent_dirs.append(comps[item][i])
      if item == 'lat_stdev':
        for i in range(comps[item].shape[0]):
          latent_stdevs.append(comps[item][i])

def load_model(output_class):
    config = Config(
    model='StyleGAN2',
    layer='style',
    output_class=output_class,
    components=80,
    use_w=True,
    batch_size=5_000, # style layer quite small
  )

    inst = get_instrumented_model(config.model, config.output_class,
                                  config.layer, torch.device('cuda'), use_w=config.use_w)

    path_to_components = get_or_compute(config, inst)

    model = inst.model

    comps = np.load(path_to_components)
    lst = comps.files
    latent_dirs = []
    latent_stdevs = []

    load_activations = False

    for item in lst:
        if load_activations:
          if item == 'act_comp':
            for i in range(comps[item].shape[0]):
              latent_dirs.append(comps[item][i])
          if item == 'act_stdev':
            for i in range(comps[item].shape[0]):
              latent_stdevs.append(comps[item][i])
        else:
          if item == 'lat_comp':
            for i in range(comps[item].shape[0]):
              latent_dirs.append(comps[item][i])
          if item == 'lat_stdev':
            for i in range(comps[item].shape[0]):
              latent_stdevs.append(comps[item][i])



app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.get('/')
async def root():
    return {'hello': 'world'}

def sample(seed: int = None, truncation: float = 0.5, w=None):
    model.truncation = truncation
    if w is None:
        w = model.sample_latent(1, seed=seed).detach().cpu().numpy()
        w = [w]*model.get_max_latents() # one per layer
  
    #save image and display
    out = model.sample_np(w)
    return Image.fromarray((out * 255).astype(np.uint8))

@app.get('/generate')
async def generate(model_type: str = 'portrait', seed: int = None, truncation: float = 0.5, w=None, save=False):
    if config.model != model_type:
        load_model(model_type)
    model.truncation = truncation
    if w is None:
        w = model.sample_latent(1, seed=seed).detach().cpu().numpy()
        w = [w]*model.get_max_latents() # one per layer
  
    #save image and display
    out = model.sample_np(w)
    final_im = Image.fromarray((out * 255).astype(np.uint8))
    
    final_im.save('/tmp/output.jpg') # save the content to temp
    
    return FileResponse('/tmp/output.jpg', media_type="image/jpeg")

class EditConfig(BaseModel):
    seed: Optional[int] = None
    model_type: Optional[str] = 'portrait'
    truncation: Optional[float] = 0.5
    start_layer: Optional[int] = 0
    end_layer: Optional[int] = 14
    attributes: Dict[str, float]
    w: Optional[str] = None

@app.post('/edit')
async def edit(edit_config: EditConfig):
    if config.model != edit_config.model_type:
        load_model(edit_config.model_type)

    seed = edit_config.seed        

    model.truncation = edit_config.truncation
    if edit_config.w is None:
        w = model.sample_latent(1, seed=seed).detach().cpu().numpy()
        w = [w]*model.get_max_latents() # one per layer
    else:
        w = [np.expand_dims(x, 0) for x in w]
    
    param_indexes = {'Gender': 1,
              'Realism': 4,
              'Gray Hair': 5,
              'Hair Length': 6,
              'Chin': 8,
              'Ponytail': 9,
              'Black Hair': 10}

    directions = []
    distances = []
    for k, v in edit_config.attributes.items():
        directions.append(latent_dirs[param_indexes[k]])
        distances.append(v)

    if directions != None and distances != None:
        for l in range(edit_config.start_layer, edit_config.end_layer):
          for i in range(len(directions)):
            w[l] = w[l] + directions[i] * distances[i] * 1
    
    torch.cuda.empty_cache()
    #save image and display
    out = model.sample_np(w)
    final_im = Image.fromarray((out * 255).astype(np.uint8))
    final_im.save('/tmp/edit_output.jpg') # save the content to temp
    
    return FileResponse('/tmp/edit_output.jpg', media_type="image/jpeg")

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)