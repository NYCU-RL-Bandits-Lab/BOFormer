# BOFormer

### Env installation 
```
conda create -n seq_qt python=3.9.7
conda activate seq_qt
conda install git pip
pip install numpy==1.23.4 (1.23.4 for RBF)
pip install platypus-opt
pip install git+https://github.com/naught101/sobol_seq@v0.2.0#egg=sobol_seq
pip install pymoo~=0.5.0
conda install -c anaconda scikit-learn
conda install ffmpeg
pip install --upgrade setuptools
conda install transformers
conda install tensorboard
conda install jupyter
pip install botorch
pip install sentencepiece
pip install yahpo_gym
conda install torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install configspace==0.6.1
pip install sentencepiece
pip install seaborn
pip install rliable

```
### Usage
```
python Q_value_Transformer/q_value_transformer.py # Traning
python Q_value_Transformer/q_value_transformer_testing_offpolicy.py # Testing
# Use data.ipynb to see the result
```

### optformer
```
pip install sentencepiece
```

### qTransformer
```
pip install transformers==4.12.1
```