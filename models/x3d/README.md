# X3D
Implementation of _"X3D: Expanding Architectures for Efficient Video Recognition" [[ArXiv](https://arxiv.org/abs/2004.04730)]_ based on the [original repository](https://github.com/facebookresearch/SlowFast).

## Install
This model has no dependencies beyond the standard dependencies found in `~/setup.py`. 

### Download weights
Four pretrained models were supplied for X3D, denotes by their sizes (XS, S, M L).
These can by downloaded by executing
```bash   
cd weights
./download_weights.sh
```

## Test 
Scripts used for testing can be found in the `scripts/test` folder. 
Detailed test results are found in the `evaluation` folder.

Model top1 (top5) accuracy % is seen in the table below:
| Model  |  1-clip       | 10-clip       | 30-clip       |  
| :---:  | :-----------: | :-----------: | :-----------: | 
| X3D-XS | 54.68 (77.52) | 65.59 (86.18) | 65.99 (86.53) |
| X3D-S  | 60.88 (82.52) | 69.97 (89.15) | 70.74 (89.48) |
| X3D-M  | 63.84 (84.27) | 72.51 (90.80) | 73.31 (91.03) |
| X3D-L  | 65.93 (85.60) | 74.37 (91.66) | 74.92 (91.99) |

NB: these results are ≈1% lower than the original results

## Profile
Scripts used for testing can be found in the `scripts/profile` folder.
Extensive profiling has been conducted on CPU (MacbookPro 16" 2019, i7), Nvidia Jetson TX2, Nvidia Jetson Xavier, and GeForce RTX 2080 Ti GPUs.







| Model  | CPU (cps)         | TX2 (cps)         | Xavier (cps)     |  RTX 2080 Ti (cps) | FLOPS (G) | Params (M) | 
| :---:  | :---------------: | :---------------: | :--------------: | :----------------: | :-------: | :--------: | 
| X3D-XS | 8.26 ± 0.11       | 8.20 ± 0.09       | 26.37 ± 0.03     | 430.15 ± 9.29      | 0.61      | 3.79       |
| X3D-S  | 2.23 ± 0.11       | 2.68 ± 0.01       | 8.07 ± 0.12      | 138.04 ± 1.69      | 1.96      | 3.79       |
| X3D-M  | 0.83 ± 0.04       | 1.47 ± 0.003      | 3.69 ± 0.02      | 55.27  ± 0.67      | 4.73      | 3.79       |
| X3D-L  | 0.25 ± 0.01       | N/A               | 0.88 ± 0.003     | 16.58  ± 0.13      | 18.37     | 6.15       |