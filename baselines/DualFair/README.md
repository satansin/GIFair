# Baseline DualFair

This folder contains the code for baseline DualFair.

### Required packages
- Python 3.8.5
- numpy == 1.22.4
- pandas == 1.1.4
- torch == 1.9.1
- scikit-learn == 0.23.2
- rdt == 0.5.3
- tqdm == 4.50.2

## Training
Training the counterfactual sample generator model with the following command.  
The trained sample generator will be saved in *./output/converter* directory.
```
python3 converter.py --dataset <DATASET> --sensitive <SENSITIVE_ATTRIBUTE> --gpu <GPU_ID>
```

Train DualFair model based on the trained counterfactual sample generator with the following command.  
Code automatically detects and loads the trained sample generator for training.  
The trained model will be saved in *./output/dualfair* directory.
```
python3 dualfair.py --dataset <DATASET> --sensitive <SENSITIVE_ATTRIBUTE> --gpu <GPU_ID>
```


## Evaluation
You can evaluate the trained model with the following command.  
Please provide a file name of the saved model checkpoint in ./output/dualfair.  
File name will have a following format - "dualfair_<DATASET>_<SENSITIVE_ATTRIBUTE>_seed_<SEED_NUM>_<TIME>"  
```
python3 evaluate.py --save_pre <FILE_NAME> --dataset <DATASET> --sensitive <SENSITIVE_ATTRIBUTE>
```
