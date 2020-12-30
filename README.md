# tetris_cem

**Joint work with [Cailin Winston](https://github.com/cailinw) and [Peter Michael](https://github.com/ptrmcl).** 
This repo contains code to design an automated Tetris player, trained using Cross Entropy Method (CEM)
optimization. The full write-up is in `tetris_cem.pdf`. This project was the response to an open-ended homework assignment in the Reinforcement Learning course at the University of Washington.

- [Overview](#overview)
- [System Requirements](#system-requirements)

# Overview
To reproduce the results, you simply run the `player.py` file, assuming hardware and software requirements are met.
```
python3 player.py
```

# System Requirements
## Hardware requirements
The code was run on an Amazon Web Services (AWS) EC2 instance, of instance type c5a.16xlarge with 64 vCPUs and 131072 MiB RAM. Generally, it is recommended to use a CPU cluster with many cores.

### Python Dependencies
The code mainly depends on the Python scientific stack.

```
numpy
scipy
gym
joblib
matplotlib
```
