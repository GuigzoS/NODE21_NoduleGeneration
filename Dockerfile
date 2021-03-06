#### FROM nvidia/cuda:11.2.0-runtime-ubuntu20.04
FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

RUN apt-get update 
RUN apt-get install ffmpeg libsm6 libxext6  -y # rajouter apt-utils
RUN apt-get install unzip
RUN apt-get -y install python3
RUN apt-get -y install python3-pip

USER algorithm

WORKDIR /opt/algorithm
ENV PATH="/home/algorithm/.local/bin:${PATH}" 
# RUN python -m pip install --user -U pip

COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
RUN python3 -m pip install --user -r requirements.txt

COPY --chown=algorithm:algorithm patch_v7nokmeans /opt/algorithm/patch_v7nokmeans
COPY --chown=algorithm:algorithm SinGAN /opt/algorithm/SinGAN

COPY --chown=algorithm:algorithm utilsGS.py /opt/algorithm/
COPY --chown=algorithm:algorithm harmonization.py /opt/algorithm/
COPY --chown=algorithm:algorithm ct_nodules_v7nokmeans.csv /opt/algorithm/

COPY --chown=algorithm:algorithm process.py /opt/algorithm/

ENTRYPOINT python3 -m process $0 $@

## ALGORITHM LABELS ##
# These labels are required
LABEL nl.diagnijmegen.rse.algorithm.name=nodulegeneration

# These labels are required and describe what kind of hardware your algorithm requires to run.
LABEL nl.diagnijmegen.rse.algorithm.hardware.cpu.count=4
LABEL nl.diagnijmegen.rse.algorithm.hardware.cpu.capabilities=()
LABEL nl.diagnijmegen.rse.algorithm.hardware.memory=40G
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.count=1
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.cuda_compute_capability=()
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.memory=15G

