# COMETH: Convex Optimization for Multiview Estimation and Tracking of Humans
## Check out our [paper](https://doi.org/10.1016/j.eswa.2026.131728) and [project page](https://parco-lab.github.io/COMETH/).


<!-- Gif area -->

<p align="center">
  <img src="static/figures/main.gif" alt="Interact-banner" width="70%">
</p>

In the era of Industry 5.0, monitoring human activity is essential for ensuring both ergonomic safety and overall well-being. We propose COMETH (Convex Optimization for Multiview Estimation and Tracking of Humans), a lightweight algorithm for real-time multi-view human pose fusion that relies on three concepts: 
- it integrates kinematic and biomechanical constraints to increase the joint positioning accuracy
- it employs convex optimization-based inverse kinematics for spatial fusion
- it implements a state observer to improve temporal consistency. 

The proposed fusion pipeline enables accurate and scalable human motion tracking, making it well-suited for industrial and safety-critical applications.

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

<p align="center">
  <img src="static/figures/pipeline.jpg" alt="Interact-banner" width="80%">
</p>

### Example
To run the complete aggregator example, please refer to [this repository](https://github.com/PARCO-LAB/multiview_hpe_aggregator):
- [`sync_dynamic_aggregator.py`](https://github.com/PARCO-LAB/multiview_hpe_aggregator/blob/master/sync_dynamic_aggregator.py)

### Compared Methods
- [Boldo et al. 2024](https://github.com/PARCO-LAB/multiview_hpe_aggregator/blob/master/sync_befine_aggregator.py)
- [OpenPTrack](https://github.com/PARCO-LAB/multiview_hpe_aggregator/blob/master/sync_openptrack_aggregator.py)

### Reference
```
@article{Martini2026,
  title = {COMETH: Convex optimization for multiview estimation and tracking of humans},
  volume = {314},
  ISSN = {0957-4174},
  url = {http://dx.doi.org/10.1016/j.eswa.2026.131728},
  DOI = {10.1016/j.eswa.2026.131728},
  journal = {Expert Systems with Applications},
  publisher = {Elsevier BV},
  author = {Martini,  Enrico and Choi,  Ho Jin and Figueroa,  Nadia and Bombieri,  Nicola},
  year = {2026},
  month = jun,
  pages = {131728}
}
```

## IMU-HPE Sensor Fusion
A sparse sensor-fusion framework for upper-limb pose estimation with shoulder-mounted IMUs and a single chest-mounted egocentric camera.

### Example
A complete example can be found in the Jupyter Notebook `IMU-HPE_fusion/test_our_model.ipynb`.

### Compared Methods
- _Li2022_ implementation: `IMU-HPE_fusion/test_li2022_model.ipynb`
- _EKF (ZeroVel)_ implementation: `IMU-HPE_fusion/test_ekf_zv_model.ipynb`

### Reference
```
Currently under review, stay tuned!
```