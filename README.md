# COMETH: Convex Optimization for Modelling, Estimating, and Tracking Humans
Biomechanical model of the human body

## Installation
Create the virtual environment (runnning Python 3.7)
```
python3.7 -m virtualenv .venv
source .venv/bin/activate
```

Install the dependences:
```
pip install torch==1.13.1
pip install nimblephysics==0.10.36
pip install cvxpy==1.2.1
pip install numpy==1.21.6
pip install pandas
pip install matplotlib
```


### Create package

```
python3 -m pip install --upgrade build
python3 -m build
```
