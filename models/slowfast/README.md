# SlowFast
_SlowFast_ [[ArXiv](https://arxiv.org/abs/1812.03982), [Repo](https://github.com/facebookresearch/SlowFast)] is two-stream 3D CNNs architecture for video-recognition. The structure includes two pathways with one pathway operating at a slower frame-rate than the other.

The code for this model is a port of [this implementation](https://github.com/facebookresearch/SlowFast).

Pretrained models can be found [here](https://github.com/facebookresearch/SlowFast/blob/master/MODEL_ZOO.md)

## Usage
See the `scripts` folder.

Other options can be explored using
```bash
python main.py --help
```