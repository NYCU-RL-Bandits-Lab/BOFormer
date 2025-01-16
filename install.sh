#!/bin/bash

# Create and activate conda environment
conda create -n boo python=3.9.7 -y
eval "$(conda shell.bash hook)"
conda activate boo

# Install git and pip
conda install git pip -y

# Install Python packages with specific versions
pip install numpy==1.26.4
pip install platypus-opt
pip install git+https://github.com/naught101/sobol_seq@v0.2.0#egg=sobol_seq
pip install pymoo~=0.5.0

# Install conda packages
conda install -c anaconda scikit-learn -y
conda install ffmpeg -y
pip install --upgrade setuptools
conda install transformers -y
conda install tensorboard -y
conda install jupyter -y

# Install additional pip packages
pip install botorch
pip install sentencepiece
pip install yahpo_gym
pip install configspace==0.6.1
pip install seaborn
pip install rliable

# Note: For PyTorch installation, uncomment the version you need
# For CUDA 12.1
conda install torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# For CUDA 11.8
# conda install torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Finally install numpy 1.23.4 for reproduce
pip install numpy==1.23.4
