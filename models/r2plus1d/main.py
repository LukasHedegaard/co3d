""" R2Plus1D main """
from ride.main import Main  # isort:skip
from ride import Configs, RideModule, TopKAccuracyMetric
from ride.optimizers import SgdOneCycleOptimizer
from torchvision.models.video.resnet import (
    BasicBlock,
    Conv2Plus1D,
    R2Plus1dStem,
    VideoResNet,
)

from datasets import ActionRecognitionDatasets


class R2Plus1D(
    RideModule,
    VideoResNet,
    ActionRecognitionDatasets,
    SgdOneCycleOptimizer,
    TopKAccuracyMetric(1, 3, 5),
):
    @staticmethod
    def configs() -> Configs:
        c = Configs()
        c.add(
            name="image_size",
            type=int,
            default=112,
            strategy="constant",
            description="Target image size. As found in the R(2+1)D paper: https://arxiv.org/abs/1711.11248",
        )
        c.add(
            name="temporal_window_size",
            type=int,
            default=8,
            strategy="choice",
            description="Temporal window size for global average pool.",
        )
        return c

    def __init__(self, hparams):
        dim_in = 3
        self.hparams.frames_per_clip = self.hparams.temporal_window_size
        self.input_shape = (dim_in, self.hparams.temporal_window_size, self.hparams.image_size, self.hparams.image_size)

        VideoResNet.__init__(
            self,
            block=BasicBlock,
            conv_makers=[Conv2Plus1D] * 4,
            layers=[2, 2, 2, 2],
            stem=R2Plus1dStem,
            num_classes=self.dataloader.num_classes,  # from ActionRecognitionDatasets
            zero_init_residual=False,
        )


if __name__ == "__main__":
    Main(R2Plus1D).argparse()
