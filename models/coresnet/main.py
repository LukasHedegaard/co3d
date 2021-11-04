""" CoX3D main """
from ride import Main  # isort:skip
from typing import Sequence

from ride import Configs, RideModule
from ride.metrics import MeanAveragePrecisionMetric, TopKAccuracyMetric
from ride.optimizers import SgdOneCycleOptimizer
from ride.utils.logging import getLogger
from ride.utils.utils import name

from datasets import ActionRecognitionDatasets
from models.coresnet.modules.coresnet import CoResNet

logger = getLogger("CoResNet")


class CoResNetRide(
    RideModule,
    ActionRecognitionDatasets,
    SgdOneCycleOptimizer,
    TopKAccuracyMetric(1),
    MeanAveragePrecisionMetric,
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
            name="co3d_temporal_fill",
            type=str,
            default="zeros",
            choices=["zeros", "replicate"],
            strategy="choice",
            description="Fill mode for samples along temporal dimension. This is used at state initialisation and in `forward_steps` as padding along the temporal dimension.",
        )
        c.add(
            name="co3d_forward_mode",
            type=str,
            default="init_frame",
            choices=["clip", "frame", "init_frame", "init_clip", "clip_init_frame"],
            strategy="choice",
            description="Whether to compute clip or frame during forward. If 'clip_init_frame', the network is initialised with a clip and then frame forwards are applied.",
        )
        c.add(
            name="co3d_num_forward_frames",
            type=int,
            default=1,
            description="The number of frames to predict over",
        )
        c.add(
            name="co3d_forward_frame_delay",
            type=int,
            default=-1,
            strategy="choice",
            description="Number of frames forwards prior to final prediction in 'clip_init_frame' mode. If '-1', a delay of clip_length - 1 is used",
        )
        c.add(
            name="co3d_forward_prediction_delay",
            type=int,
            default=0,
            strategy="choice",
            description="Number of steps to delay the prediction relative to the frames",
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
        # Inflate frames_per_clip of dataset, to have some frames for initialization
        self.temporal_window_size = self.hparams.frames_per_clip
        image_size = self.hparams.image_size
        dim_in = 3

        self.module = CoResNet(
            arch=self.hparams.resnet_architecture,
            dim_in=dim_in,
            image_size=image_size,
            frames_per_clip=self.temporal_window_size,
            num_classes=self.dataloader.num_classes,  # from ActionRecognitionDatasets,
            resnet_depth=self.hparams.resnet_depth,
            resnet_num_groups=self.hparams.resnet_num_groups,
            resnet_width_per_group=self.hparams.resnet_width_per_group,
            resnet_dropout_rate=self.hparams.resnet_dropout_rate,
            resnet_fc_std_init=self.hparams.resnet_fc_std_init,
            resnet_final_batchnorm_zero_init=self.hparams.resnet_final_batchnorm_zero_init,
            resnet_head_act=self.hparams.resnet_head_act,
            enable_detection=False,
            align_detection=False,
            temporal_fill=self.hparams.co3d_temporal_fill,
        )

        if "frame" in self.hparams.co3d_forward_mode:
            self.module.call_mode = "forward_steps"

            num_init_frames = max(
                self.module.receptive_field - 1,
                self.hparams.co3d_forward_frame_delay - 1,
            )
            self.hparams.frames_per_clip = (
                num_init_frames
                + self.hparams.co3d_num_forward_frames
                + self.hparams.co3d_forward_prediction_delay
            )

        # From ActionRecognitionDatasets
        frames_per_clip = (
            1
            if self.hparams.co3d_forward_mode == "frame"
            else self.hparams.frames_per_clip
        )
        self.input_shape = (dim_in, frames_per_clip, image_size, image_size)

        # Assuming Conv3d have stride = 1 and dilation = 1, and that no other modules delay the network.
        logger.info(f"Model receptive field: {self.module.receptive_field} frames")
        logger.info(f"Loss: {name(self.loss)}")

    def map_loaded_weights(self, finetune_from_weights, state_dict):
        # Map state_dict for "Slow" weights
        state_dict = {
            (
                k.replace("blocks", "module")
                .replace("res_module.", "pathway0_res")
                .replace("branch1_conv", "0.0.branch1")
                .replace("branch1_norm", "0.0.branch1_bn")
                .replace("branch2", "0.1.branch2")
                .replace("3.pathway0_res0.0.0.", "3.pathway0_res0.0.0.0.")
                .replace("4.pathway0_res0.0.0.", "4.pathway0_res0.0.0.0.")
                .replace("proj", "projection")
            ): v
            for k, v in state_dict.items()
        }
        return state_dict

    def clean_state_on_shape_change(self, shape):
        if getattr(self, "_current_input_shape", None) != shape:
            self.module.clean_state()
            self._current_input_shape = shape

    def forward(self, x):
        if not hasattr(self.hparams, "profile"):
            self.module.clean_state()

        result = None

        if "init" in self.hparams.co3d_forward_mode:
            self.warm_up(tuple(x[:, :, 0].shape))

            num_init_frames = max(
                self.module.receptive_field - 1,
                self.hparams.co3d_forward_frame_delay - 1,
            )
            self.module(x[:, :, :num_init_frames])  # = forward_steps don't save

            result = self.module(x[:, :, num_init_frames:])
        else:
            result = self.module(x)

        if self.task == "classification":
            result = result.mean(dim=-1)
        elif self.task == "detection":
            result = result.permute(0, 2, 1).reshape(-1, self.num_classes)
        else:
            raise ValueError(f"Unknown task {self.task}")

        return result

    def warm_up(self, input_shape: Sequence[int], *args, **kwargs):
        for m in self.module.modules():
            if hasattr(m, "state_index"):
                m.state_index = 0


if __name__ == "__main__":
    Main(CoResNetRide).argparse()
