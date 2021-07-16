FROM nvidia/cuda:10.0-base-ubuntu18.04

MAINTAINER Adrian Campos "https://github.com/adriancampos"


# install Python
ARG _PY_SUFFIX=3
ARG PYTHON=python${_PY_SUFFIX}
ARG PIP=pip${_PY_SUFFIX}

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y \
    ${PYTHON} \
    ${PYTHON}-pip

RUN ${PIP} --no-cache-dir install --upgrade \
    pip \
    setuptools
	
RUN ln -s $(which ${PYTHON}) /usr/local/bin/python 



RUN mkdir -p /app

WORKDIR /app

COPY . .

COPY requirements.txt .

RUN pip install -r requirements.txt \

RUN python -c "import nltk; nltk.download('wordnet')"

ENV CUDA_HOME=/usr/local/cuda-10.1

RUN cd iPERCore && python setup.py develop


EXPOSE 8000

CMD ["python", "main.py"]