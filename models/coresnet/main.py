""" CoX3D main """
from ride import Main  # isort:skip

from ride import Configs, RideModule
from ride.metrics import MeanAveragePrecisionMetric, MetricSelector, TopKAccuracyMetric
from ride.optimizers import SgdOneCycleOptimizer
from ride.utils.logging import getLogger

from datasets import ActionRecognitionDatasets
from datasets.ava import AvaMetric, ava_loss
from models.common import Co3dBase
from models.coresnet.modules.coresnet import CoResNet

logger = getLogger("CoResNet")


class CoResNetRide(
    RideModule,
    Co3dBase,
    ActionRecognitionDatasets,
    SgdOneCycleOptimizer,
    MetricSelector(
        kinetics400=TopKAccuracyMetric(1),
        charades=MeanAveragePrecisionMetric,
        ava=AvaMetric,
        default_config="ava",
    ),
):
    @staticmethod
    def configs() -> Configs:
        c = Configs()
        c.add(
            name="resnet_architecture",
            type=str,
            default="slow",
            strategy="constant",
            choices=["slow", "i3d"],
            description="Architecture of ResNet.",
        )
        c.add(
            name="resnet_depth",
            type=int,
            default=50,
            strategy="constant",
            choices=[50, 101],
            description="Number of layers in ResNet.",
        )
        c.add(
            name="resnet_num_groups",
            type=int,
            default=1,
            strategy="constant",
            description="Number of groups.",
        )
        c.add(
            name="resnet_width_per_group",
            type=int,
            default=64,
            strategy="constant",
            description="Width of each group.",
        )
        c.add(
            name="resnet_dropout_rate",
            type=float,
            default=0.5,
            choices=[0.0, 1.0],
            strategy="uniform",
            description="Dropout rate before final projection in the backbone.",
        )
        c.add(
            name="resnet_head_act",
            type=str,
            default="softmax",
            choices=["softmax", "sigmoid"],
            strategy="choice",
            description="Activation layer for the output head.",
        )
        c.add(
            name="resnet_fc_std_init",
            type=float,
            default=0.01,
            strategy="choice",
            description="The std to initialize the fc layer(s).",
        )
        c.add(
            name="resnet_final_batchnorm_zero_init",
            type=int,
            default=1,
            choices=[0, 1],
            strategy="choice",
            description="If true, initialize the gamma of the final BN of each block to zero.",
        )
        c.add(
            name="enable_detection",
            type=int,
            default=0,
            strategy="choice",
            choices=[0, 1],
            description="Whether to enable detection head.",
        )
        c.add(
            name="align_detection",
            type=int,
            default=0,
            strategy="choice",
            choices=[0, 1],
            description="Whether to utilise alignment in detection head.",
        )
        return c

    def __init__(self, hparams):
        if "ava" == self.hparams.dataset:
            self.loss = ava_loss(self.loss)
            self.hparams.enable_detection = True

        self.module = CoResNet(
            arch=self.hparams.resnet_architecture,
            dim_in=self.dim_in,
            image_size=self.hparams.image_size,
            temporal_window_size=self.hparams.temporal_window_size,
            num_classes=self.dataloader.num_classes,  # from ActionRecognitionDatasets,
            resnet_depth=self.hparams.resnet_depth,
            resnet_num_groups=self.hparams.resnet_num_groups,
            resnet_width_per_group=self.hparams.resnet_width_per_group,
            resnet_dropout_rate=self.hparams.resnet_dropout_rate,
            resnet_fc_std_init=self.hparams.resnet_fc_std_init,
            resnet_final_batchnorm_zero_init=self.hparams.resnet_final_batchnorm_zero_init,
            resnet_head_act=self.hparams.resnet_head_act,
            enable_detection=self.hparams.enable_detection,
            align_detection=self.hparams.align_detection,
            temporal_fill=self.hparams.co3d_temporal_fill,
        )

    def forward(self, x):
        if self.hparams.enable_detection:
            # Pass in bounding boxes to RoIHead.
            self.module[-1].set_boxes(x["boxes"])
            x = x["images"]
        return Co3dBase.forward(self, x)

    def map_loaded_weights(self, finetune_from_weights, state_dict):
        # Map state_dict for "Slow" weights
        state_dict = {
            (
                k.replace("model.", "")
                .replace("blocks", "module")
                .replace("res_module.", "pathway0_res")
                .replace("branch1_conv", "0.0.branch1")
                .replace("branch1_norm", "0.0.branch1_bn")
                .replace("branch2", "0.1.branch2")
                .replace("3.pathway0_res0.0.0.", "3.pathway0_res0.0.0.0.")
                .replace("4.pathway0_res0.0.0.", "4.pathway0_res0.0.0.0.")
                .replace("detection_head", "module.5")
                .replace("proj", "projection")
            ): v
            for k, v in state_dict.items()
        }
        return state_dict


if __name__ == "__main__":
    Main(CoResNetRide).argparse()
