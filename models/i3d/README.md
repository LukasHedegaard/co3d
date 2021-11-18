# I3D
_I3D_ [[ArXiv](https://arxiv.org/abs/1705.07750), [Repo](https://github.com/deepmind/kinetics-i3d)] is a 3D CNN for activity recognition, proposed to "inflate" the weights from a 2D CNN pretrained on ImageNet in the initialisation of the 3D CNN, thereby improving accuracy and reducing training time.

The code for this model is a port of [this implementation](https://github.com/facebookresearch/SlowFast).

Pretrained models can be found [here](https://github.com/facebookresearch/SlowFast/blob/master/MODEL_ZOO.md)

## Usage
See the `scripts` folder.

Other options can be explored using
```bash
python main.py --help
```