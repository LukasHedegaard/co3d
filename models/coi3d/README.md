# Continual I3D
Continual implementation of Inflated 3D ConvNet [[ArXiv](https://arxiv.org/abs/1705.07750)]

The code for this model is a port of [this implementation](https://github.com/facebookresearch/SlowFast) modified to work as a continual inference network.

Pretrained models can be found [here](https://pytorchvideo.readthedocs.io/en/latest/model_zoo.html).

The code can execute both the regular clip-based inference as fell as the frame-based one.

## Usage
See the `scripts` folder for usage examples.

Other options can be explored using
```bash
python main.py --help
```
