# Code for GIFair
This folder contains the code for the our proposed method GIFair.

## Installation
### Requirements
- Python >= 3.6
- [PyTorch](https://pytorch.org/) >= 1.4.0
- `pip install -r requirements.txt`

## Run GIFair
To run GIFair, you may need to run the two python programs, [`main.py`](main.py) and [`evaluate.py`](evaluate.py), for training and evaluation, respectively. By default, both training and evaluation will utilize the parameters specified in the configuration file [`config.json`](config.py). Normally, you may need run GIFair with particular parameters (e.g., group-fair coefficient, individual-fair coefficient, etc.).

### Run with Specified Parameters
The following command gives an example of runnning GIFair training on Adult income dataset with group-fair coefficient 1 and individual-fair coefficient 2.
```bash
python main.py --fair-coeff 1 --fair-coeff-individual 2 --dataset adult
```
Note that the coefficient for accuracy is fixed to 1. Other coefficients are default to 1. If a non-zero gamma is specified, the focal loss function will be enabled. More default options for each dataset is shown in [`config.json`](config.py).

You can also see more detailed options from
```bash
python main.py -h
```
Result files for training will be saved in `results/`. Saved models will be saved in `saved/`.

To evaluate the trained model with specific parameters (e.g., those involved in the above mentioned example), use the follwing command.
```bash
python evaluate.py --fair-coeff 1 --fair-coeff-individual 2 --dataset adult
```
The evaluation result will be saved in `evaluated/`. One example of the evaluation result file is shown as follows.
```json
{
    "test": {
        "YNN": 0.9677025232403719,
        "acc": 0.850398406374502,
        "precision": 0.739966832504146,
        "recall": 0.6029729729729729,
        "F1": 0.664482501861504,
        "DP": 0.11617849808105801,
        "EO": 0.04264716251923291
    }
}
```
In this example, we can easily read the value of each measurement used in our experiments.

### Run All Experiments

To run all our experiment instances, you could directly execute our script files [`linux_gifair.sh`](linux_gifair.sh) (for UNIX-based system) or [`win_gifair.bat`](win_gifair.bat) (for Window).
