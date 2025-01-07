# Decision Transformer

## Env configuration
DT needs to run under transformers==4.12.1 with our environment.

## Offline Data Generation
The offline data are 100 trajectories generated from qEHVI and random policy (rate 1:1).
Generate Offline Data:
```shell
python Decision_Transformer/decision_transformer/envs/generate_bo_data.py --f_num 2
python Decision_Transformer/decision_transformer/envs/generate_bo_data.py --f_num 3
```
Or download data from: https://drive.google.com/file/d/1V4Hl939ikB3SEq0m6QmIqlvSjcGisyUi/view?usp=sharing
Put data in directory: Decision_Transformer/decision_transformer/envs/data
Pretrained Model stored in Decision_Transformer/preTrained

## Training
```shell
python Decision_Transformer/experiment.py --f_num 2
python Decision_Transformer/experiment.py --f_num 3
```

## Testing
```shell
python Decision_Transformer/experiment.py --f_num 2 --function_type AR
python Decision_Transformer/experiment.py --f_num 3 --function_type ARS
```