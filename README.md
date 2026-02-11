# COMETH: Convex Optimization for Modelling, Estimating, and Tracking Humans
Biomechanical model of the human body

## Installation
Create the virtual environment (runnning Python 3.7):
```
python3.7 -m virtualenv .venv
source .venv/bin/activate
```

Install the dependences:
```
pip install -m requirements.txt
```

Create the COMETH package:
```
pip install -e .
```

# COMETH Applications

## Multiview Skeleton Fusion
A novel convex optimization-based framework for real-time multi-view human pose fusion.

### Example
To run the complete aggregator example, please refer to [this repository](https://github.com/PARCO-LAB/multiview_hpe_aggregator):
- [`sync_dynamic_aggregator.py`](https://github.com/PARCO-LAB/multiview_hpe_aggregator/blob/master/sync_dynamic_aggregator.py)

### Compared Methods
- [Boldo et al. 2024](https://github.com/PARCO-LAB/multiview_hpe_aggregator/blob/master/sync_befine_aggregator.py)
- [OpenPTrack](https://github.com/PARCO-LAB/multiview_hpe_aggregator/blob/master/sync_openptrack_aggregator.py)

### Reference
```
@article{martini2025cometh,
  title={COMETH: Convex Optimization for Multiview Estimation and Tracking of Humans},
  author={Martini, Enrico and Choi, Ho Jin and Figueroa, Nadia and Bombieri, Nicola},
  doi={https://doi.org/10.48550/arXiv.2508.20920}
  journal={arXiv preprint arXiv:2508.20920},
  year={2025},
}
```

## IMU-HPE Sensor Fusion
A sparse sensor-fusion framework for upper-limb pose estimation with shoulder-mounted IMUs and a single chest-mounted egocentric camera.

### Example
A complete example can be found in the Jupyter Notebook `IMU-HPE_fusion/test_our_model.ipynb`.

### Compared Methods
- _Li2022_ implementation: `IMU-HPE_fusion/test_li2022_model.ipynb`
- _EKF (ZeroVel)_ implementation: `IMU-HPE_fusion/test_ekf_zv_model.ipynb`