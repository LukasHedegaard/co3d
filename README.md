# Continual 3D Convolutional Neural Networks
[![Paper](http://img.shields.io/badge/paper-arxiv.2106.00050-B31B1B.svg)](https://arxiv.org/abs/2106.00050)
[![Framework](https://img.shields.io/badge/Built_to-Ride-643DD9.svg)](https://github.com/LukasHedegaard/ride)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Continual 3D Convolutional Neural Networks (Co3D CNNs) are a new computational formulation of spatio-temporal 3D CNNs, in which videos are processed frame-by-frame rather than by clip.

In online processing tasks demanding frame-wise predictions, Co3D CNNs dispense with the computational redundancies of regular 3D CNNs, namely the repeated convolutions over frames, which appear in multiple clips.

Co3D CNNs are weight-compatible with regular 3D CNNs, do not need further training, and reduce the floating point operations for frame-wise computations by more than an order of magnitude!

## News
- 2022-07-04 Our paper _"Continual 3D Convolutional Neural Networks for Real-time Processing of Videos"_ has been accepted at the [European Conference on Computer Vision (ECCV) 2022](https://eccv2022.ecva.net).


## Principle 

<div align="center">
  <img src="figures/coconv.png" width="500">
  <br>
  Continual Convolution. 
	An input (green d or e) is convolved with a kernel (blue α, β). The intermediary feature-maps corresponding to all but the last temporal position are stored, while the last feature map and prior memory are summed to produce the resulting output. For a continual stream of inputs, Continual Convolutions produce identical outputs to regular convolutions.
</div>


## Results
<div align="center">
  <img src="figures/acc-vs-flops.png" width="500">
  <br>
  Accuracy/complexity trade-off for Continual X3D CoX3D and recent state-of-the-art 3D CNNs on Kinetics-400 using 1-clip/frame testing. 
  For regular 3D CNNs, the FLOPs per clip ■ are noted, while the FLOPs per frame ● are shown for the Continual 3D CNNs. 
  The CoX3D models used the weights from the X3D models without further fine-tuning.
  The global average pool size for the network is noted in each point.
  The diagonal and vertical arrows indicate respectively a transfer from regular to Continual 3D CNN and an extension of receptive field.

  <br>
  <br>

<img src="figures/results.png">
<br>
  Benchmark of state-of-the-art methods on Kinetics-400. The noted accuracy is the single clip or frame top-1 score using RGB as the only input-modality. 
  The performance was evaluated using publicly available pre-trained models without any further fine-tuning.
  For thoughput comparison, evaluations per second denote frames per second for the CoX3D models and clips per second for the remaining models. Throughput results are the mean +/- std of 100 measurements. Pareto-optimal models are marked with bold. Mem. is the maximum allocated memory during inference noted in megabytes.
</div>



# Setup

1. Clone the project code
    ```bash
    git clone https://github.com/LukasHedegaard/co3d
    cd co3d
    ```

1. Create and activate `conda` environent (optional)
    ```bash
    conda create --name co3d python=3.8
    conda activate co3d
    ```

1. Install Python dependencies
    ```bash
    pip install -e .[dev]
    ``` 

1. Install [FFMPEG](https://ffmpeg.org) and [UNRAR](https://www.rarlab.com/rar_add.htm)

1. Fill in the information on your dataset folder path in the `.env` file:
    ```bash
    DATASETS_PATH=/path/to/datasets
    LOGS_PATH=/path/to/logs
    CACHE_PATH=.cache
    ```

1. Download dataset using [these instructions](datasets/README.md)


# Models

## [CoX3D](models/cox3d/README.md)
_CoX3D_ is the Continual-CNN implementation of X3D.
In contrast to regular 3D CNNs, which take a whole video clip as input, Continual CNNs operate frame-by-frame and can thus speed up computation by a significant margin.


## [CoSlow](models/coslow/README.md)
_CoSlow_ is the Continual-CNN implementation of Slow.


## [CoI3D](models/coi3d/README.md)
_CoSlow_ is the Continual-CNN implementation of I3d.


## [X3D](models/x3d/README.md)
_X3D_ [[ArXiv](https://arxiv.org/abs/2004.04730), [Repo](https://github.com/facebookresearch/SlowFast)] is a family of 3D variants of the EfficientNet achitecture, which produce state-of-the-art results for lightweight human activity recognition.


## [R(2+1)D](models/r2plus1d/README.md)
_R(2+1)D_ [[ArXiv](https://arxiv.org/abs/1705.07750), [Repo](https://pytorch.org/vision/stable/models.html#torchvision.models.video.r2plus1d_18)] is a CNN for activity recognition, which separates the 3D convolution into a spatial 2D convolution and a temporal 1D convolution in order to reduce the number of parameters and increase the network efficiency.


## [I3D](models/i3d/README.md)
_I3D_ [[ArXiv](https://arxiv.org/abs/1705.07750), [Repo](https://github.com/deepmind/kinetics-i3d)] is a 3D CNN for activity recognition, proposed to "inflate" the weights from a 2D CNN pretrained on ImageNet in the initialisation of the 3D CNN, thereby improving accuracy and reducing training time.

The implementation here is a port of the one found in the [SlowFast Repo](https://github.com/facebookresearch/SlowFast).


## [SlowFast](models/slowfast/README.md)
_SlowFast_ [[ArXiv](https://arxiv.org/abs/1812.03982), [Repo](https://github.com/facebookresearch/SlowFast)] is two-stream 3D CNNs architecture for video-recognition. The structure includes two pathways with one pathway operating at a slower frame-rate than the other.


## [Slow](models/coresnet/README.md)
_Slow_ is the "slow" branch of the SlowFast network [[ArXiv](https://arxiv.org/abs/1812.03982), [Repo](https://github.com/facebookresearch/SlowFast)]

# Usage
The project code written in PyTorch and uses [Ride](https://github.com/LukasHedegaard/ride) to provide implementations of training, evaluations, and benchmarking methods.
A plethora of usage options are available, which are best explored in the [Ride docs](https://ride.readthedocs.io) or the command-line help, e.g.:
```bash
python models/cox3d/main.py --help 
```

This repository contains the implementations of Continual X3D (CoX3D), as well as number of 3D-CNN baselines.

Each model has its own folder with a self-contained implementation, scripts, weight download utilities, hparams and profiling results. 
Overview tables for scripts used to download weights, run the model test-sequences, and throughput benchmarks are found below:

## Download weights
| Model         | Dataset  | Download |
| -------       | -------- | -------- |
| I3D-R50       | Kinetics | [download](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/I3D_8x8_R50.pkl)
| R(2+1)D-18    | Kinetics | [download](https://download.pytorch.org/models/r2plus1d_18-91a641e6.pth)
| SlowFast-8x8  | Kinetics | [download](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWFAST_8x8_R50.pkl)
| SlowFast-4x16 | Kinetics | [download](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWFAST_4x16_R50.pkl)
| Slow-8x8      | Kinetics | [download](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/SLOW_8x8_R50.pyth)
| (Co)X3D-XS    | Kinetics | [download](https://dl.fbaipublicfiles.com/pyslowfast/x3d_models/x3d_xs.pyth)
| (Co)X3D-S     | Kinetics | [download](https://dl.fbaipublicfiles.com/pyslowfast/x3d_models/x3d_s.pyth)
| (Co)X3D-M     | Kinetics | [download](https://dl.fbaipublicfiles.com/pyslowfast/x3d_models/x3d_m.pyth)
| (Co)X3D-L     | Kinetics | [download](https://dl.fbaipublicfiles.com/pyslowfast/x3d_models/x3d_l.pyth)
| (Co)Slow-8x8  | Charades | [download](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/charades/SLOW_8x8_R50.pyth)


## Evaluate on Kinetics400
Evaluate the 1-clip accuracy of pretrained models. 
The scripts should be executed from project root.

| Model         | Script |
| -------       | -------- | 
| I3D-R50       | [`./models/i3d/scripts/test/kinetics400.sh`](models/i3d/scripts/test/kinetics400.sh) | 
| R(2+1)D-18    | [`./models/r2plus1d/scripts/test/kinetics400.sh`](models/r2plus1d/scripts/test/kinetics400.sh) | 
| SlowFast      | [`./models/slowfast/scripts/test/kinetics400.sh`](models/slowfast/scripts/test/kinetics400.sh) | 
| Slow          | [`./models/slow/scripts/test/kinetics400.sh`](models/slow/scripts/test/kinetics400.sh) | 
| X3D           | [`./models/x3d/scripts/test/kinetics400.sh`](models/x3d/scripts/test/kinetics400.sh) | 
| CoX3D         | [`./models/cox3d/scripts/test/kinetics400.sh`](models/cox3d/scripts/test/kinetics400.sh) | 
| CoSlow        | [`./models/coslow/scripts/test/kinetics400.sh`](models/coslow/scripts/test/kinetics400.sh) | 
| CoI3D         | [`./models/coi3d/scripts/test/kinetics400.sh`](models/coi3d/scripts/test/kinetics400.sh) | 


## Evaluate on Charades
Evaluate the 1-clip accuracy of pretrained models. 
The scripts should be executed from project root.

| Model         | Script |
| -------       | -------- | 
| (Co)Slow-8x8       | [`./models/coslow/scripts/test/charades.sh`](models/coslow/scripts/test/charades.sh) | 


## Benchmark FLOPs and throughput
The scripts should be executed from project root.

| Model         | Script |
| -------       | -------- | 
| I3D-R50       | [`./models/i3d/scripts/profile/kinetics400.sh`](models/i3d/scripts/profile/kinetics400.sh) | 
| R(2+1)D-18    | [`./models/r2plus1d/scripts/profile/kinetics400.sh`](models/r2plus1d/scripts/profile/kinetics400.sh) | 
| SlowFast      | [`./models/slowfast/scripts/profile/kinetics400.sh`](models/slowfast/scripts/profile/kinetics400.sh) | 
| Slow          | [`./models/slow/scripts/profile/kinetics400.sh`](models/slow/scripts/profile/kinetics400.sh) | 
| X3D           | [`./models/x3d/scripts/profile/kinetics400.sh`](models/x3d/scripts/profile/kinetics400.sh) | 
| CoX3D         | [`./models/cox3d/scripts/profile/kinetics400.sh`](models/cox3d/scripts/profile/kinetics400.sh) | 
| CoI3D         | [`./models/coi3d/scripts/profile/kinetics400.sh`](models/coi3d/scripts/profile/kinetics400.sh) | 
| CoSlow        | [`./models/coslow/scripts/profile/kinetics400.sh`](models/coslow/scripts/profile/kinetics400.sh) | 


# Citation   
```
@inproceedings{hedegaard2022continual,
    title={Continual 3D Convolutional Neural Networks for Real-time Processing of Videos},
    author={Lukas Hedegaard and Alexandros Iosifidis},
    booktitle={European Conference on Computer Vision (ECCV)},
    year={2022},
}
```

## Acknowledgement
This work has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 871449 (OpenDR).
