# Adversarial Learning of Group and Individual Fair Representations (GIFair)
This is the code repository of our paper [Adversarial Learning of Group and Individual Fair Representations](https://link.springer.com/chapter/10.1007/978-981-97-2242-6_15). Please find the technical version of the paper containing the supplementary materials here.

## Installation
### Requirements
- Python >= 3.6
- [PyTorch](https://pytorch.org/) >= 1.4.0
- `pip install -r requirements.txt`

## Run GIFair
### Run with Specified Parameters
Run GIFair training on Adult income dataset with group-fair coefficient 1 and individual-fair coefficient 2.
```bash
python main.py --fair-coeff 1 --fair-coeff-individual 2 --dataset adult
```
Note that the coefficient for accuracy is fixed to 1. Other coefficients are default to 1. If a non-zero gamma is specified, the focal loss function will be enabled. More default options for each dataset is shown in `config.json`.

You can see more detailed options from
```bash
python main.py -h
```
Result files for training will be saved in `results/`. Saved models will be saved in `saved/`.

To evaluate the trained model with specific parameters, using the follwing command.
```bash
python evaluate.py --fair-coeff 1 --fair-coeff-individual 2 --dataset adult
```
