# Baselines UNFAIR, ALFR, LAFTR, DCFR and iFair
Source code for baselines **UNFAIR**, **ALFR**, **LAFTR**, **DCFR**, **iFair**, which use the same code framework.

## Installation
### Requirements
- Linux with Python >= 3.6
- [PyTorch](https://pytorch.org/) >= 1.4.0
- `pip install -r requirements.txt`

## Quick Start
### Run for single fair coefficient and random seed
You can run the code with specific parameters for a particular model.
For example, the below command runs DCFR model on Adult income dataset and conditional fairness task with fair coefficient 20.
```bash
python main.py --model DCFR --task CF --dataset adult --seed 0 --fair-coeff 20
```
Substitude the model name with UNFAIR, ALFR, LAFTR or IFAIR to run different models.

You can see more options from
```bash
python main.py -h
```
Result files will be saved in `results/`. Saved models will be saved in `saved/`. Tensorboard logs will be saved in `tensorboard/`.

### Run all experiments
To run all our experiment instances, you could directly execute our script files [`linux_test.sh`](linux_test.sh) (for UNIX-based system) or [`win_test.bat`](win_test.bat) (for Window).
