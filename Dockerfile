FROM 812206152185.dkr.ecr.us-west-2.amazonaws.com/latch-base:02ab-main

# Install dependencies

RUN python3 -m pip install -U pip &&\
    python3 -m pip install -U matplotlib jupyterlab

# Install NUPACK

## Get NUPACK via curl and unzip
RUN curl -L https://github.com/beliveau-lab/NUPACK/archive/refs/tags/v4.0.0.23.zip -o nupack-4.0.0.23.zip &&\
    unzip nupack-4.0.0.23.zip
## Install with pip
RUN python3 -m pip install -U nupack -f /root/NUPACK-4.0.0.23/src/package


# The following lines are needed to ensure your build environement works correctly with latch.

COPY wf /root/wf
RUN  sed -i 's/latch/wf/g' flytekit.config
ARG tag
ENV FLYTE_INTERNAL_IMAGE $tag
RUN python3 -m pip install --upgrade latch
WORKDIR /root