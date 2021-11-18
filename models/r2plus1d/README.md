# R(2+1)D
_R(2+1)D_ [[ArXiv](https://arxiv.org/abs/1705.07750), [Repo](https://pytorch.org/vision/stable/models.html#torchvision.models.video.r2plus1d_18)] is a CNN for activity recognition, which separates the 3D convolution into a spatial 2D convolution and a temporal 1D convolution in order to reduce the number of parameters and increase the network efficiency.

The code for this model is a port of [this implementation](https://github.com/facebookresearch/SlowFast).

Pretrained models can be found [here](https://github.com/facebookresearch/SlowFast/blob/master/MODEL_ZOO.md)

## Usage
See the `scripts` folder.

Other options can be explored using
```bash
python main.py --help
```