# Baseline LFR
Source code for baseline **LFR**.
We mainly follow the python implementation of
[Zemel et al. 2013](http://www.cs.toronto.edu/~toni/Papers/icml-final.pdf).

## Requirements
Install the requirement packages with the following commands:
```bash
conda create --name lfr python=3.8
conda activate lfr
conda install -c anaconda scipy
conda install -c conda-forge numba=0.48.0
conda install -c conda-forge scikit-learn
conda install -c conda-forge pandas
```

## Run with 
You can run the code with specific parameters for a particular model:
- First parameter: dataset name
- Second parameter: group fairness coefficient
- Third parameter: individual fairness coefficient
- Fourth parameter: random seed
For example, the below command runs LFR on Adult income dataset with group fairness coefficient 1,
individual fairness coefficient 2 and random seed 0.
```bash
python lfr.py adult 1 2 0
```

### Run all experiments
To run all our experiment instances, you could directly execute our script files
[`run.sh`](run.sh) (for UNIX-based system) or [`run.bat`](run.bat) (for Window).
