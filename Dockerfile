FROM ubuntu:20.04

RUN apt-get update && apt-get install -y \
	git wget screen tmux bzip2 gcc

# paths
RUN mkdir /workspace
RUN mkdir /workspace/embeddings

RUN mkdir /mnt/data
RUN mkdir /mnt/pred

ADD scripts /workspace/
ADD embeddings /workspace/embeddings/

RUN chmod +x /workspace/*.sh

# miniconda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
RUN chmod +x miniconda.sh
RUN ./miniconda.sh -b -p /miniconda3
RUN chmod -R 777 /miniconda3
RUN rm ./miniconda.sh
ENV PATH="/miniconda3/bin:${PATH}"
RUN conda install -y python=3.10

RUN pip install numpy
RUN pip install nibabel
RUN pip install tqdm
RUN pip install scipy
RUN pip install PyWavelets


