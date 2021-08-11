""" CoX3D main """
from ride import Main  # isort:skip
from ride import Configs, RideModule
from ride.metrics import TopKAccuracyMetric
from ride.optimizers import SgdCyclicLrOptimizer
from ride.utils.logging import getLogger

from datasets import ActionRecognitionDatasets
from models.cox3d.modules.x3d import CoX3D

logger = getLogger("CoX3D")


class CoX3DRide(
    RideModule,
    CoX3D,
    ActionRecognitionDatasets,
    SgdCyclicLrOptimizer,
    TopKAccuracyMetric(1, 3, 5),
):
    @staticmethod
    def configs() -> Configs:
        c = Configs()
        c.add(
            name="x3d_num_groups",
            type=int,
            default=1,
            strategy="constant",
            description="Number of groups.",
        )
        c.add(
            name="x3d_width_per_group",
            type=int,
            default=64,
            strategy="constant",
            description="Width of each group.",
        )
        c.add(
            name="x3d_width_factor",
            type=float,
            default=1.0,
            strategy="constant",
            description="Width expansion factor.",
        )
        c.add(
            name="x3d_depth_factor",
            type=float,
            default=1.0,
            strategy="constant",
            description="Depth expansion factor.",
        )
        c.add(
            name="x3d_bottleneck_factor",
            type=float,
            default=1.0,
            strategy="constant",
            description="Bottleneck expansion factor for the 3x3x3 conv.",
        )
        c.add(
            name="x3d_conv1_dim",
            type=int,
            default=12,
            strategy="constant",
            description="Dimensions of the first 3x3 conv layer.",
        )
        c.add(
            name="x3d_conv5_dim",
            type=int,
            default=2048,
            strategy="constant",
            description="Dimensions of the last linear layer before classificaiton.",
        )
        c.add(
            name="x3d_use_channelwise_3x3x3",
            type=int,
            default=1,
            choices=[0, 1],
            strategy="choice",
            description="Whether to use channelwise (=depthwise) convolution in the center (3x3x3) convolution operation of the residual blocks.",
        )
        c.add(
            name="x3d_dropout_rate",
            type=float,
            default=0.5,
            choices=[0.0, 1.0],
            strategy="uniform",
            description="Dropout rate before final projection in the backbone.",
        )
        c.add(
            name="x3d_head_activation",
            type=str,
            default="softmax",
            choices=["softmax", "sigmoid"],
            strategy="choice",
            description="Activation layer for the output head.",
        )
        c.add(
            name="x3d_head_batchnorm",
            type=int,
            default=0,
            choices=[0, 1],
            strategy="choice",
            description="Whether to use a BatchNorm layer before the classifier.",
        )
        c.add(
            name="x3d_fc_std_init",
            type=float,
            default=0.01,
            strategy="choice",
            description="The std to initialize the fc layer(s).",
        )
        c.add(
            name="x3d_final_batchnorm_zero_init",
            type=int,
            default=1,
            choices=[0, 1],
            strategy="choice",
            description="If true, initialize the gamma of the final BN of each block to zero.",
        )
        c.add(
            name="co3d_temporal_fill",
            type=str,
            default="replicate",
            choices=["zeros", "replicate"],
            strategy="choice",
            description="Fill mode for samples along temporal dimension. This is used at state initialisation and in `forward3d` as padding along the temporal dimension.",
        )
        c.add(
            name="co3d_forward_mode",
            type=str,
            default="frame",
            choices=["clip", "frame", "init_frame", "clip_init_frame"],
            strategy="choice",
            description="Whether to compute clip or frame during forward. If 'clip_init_frame', the network is initialised with a clip and then frame forwards are applied.",
        )
        c.add(
            name="co3d_num_forward_frames",
            type=int,
            default=1,
            description="The number of frames to forward and average over. This is unused for `co3d_forward_mode='clip'`.",
        )
        c.add(
            name="co3d_forward_frame_delay",
            type=int,
            default=-1,
            strategy="choice",
            description="Number of frame forwards prior to final prediction in 'clip_init_frame' mode. If '-1', a delay of clip_length - 1 is used",
        )

        return c

    def __init__(self, hparams):
        # Inflate frames_per_clip of dataset, to have some frames for initialization
        self.temporal_window_size = self.hparams.frames_per_clip

        if "clip" not in self.hparams.co3d_forward_mode:
            self.hparams.frames_per_clip = 0

        if "frame" in self.hparams.co3d_forward_mode:
            assert self.hparams.co3d_num_forward_frames > 0
            self.hparams.frames_per_clip += self.hparams.co3d_num_forward_frames

        if "init" in self.hparams.co3d_forward_mode:
            if self.hparams.co3d_forward_frame_delay < 0:
                self.hparams.co3d_forward_frame_delay = (
                    self.temporal_window_size
                    + 1
                    + self.hparams.co3d_forward_frame_delay
                )
            self.hparams.frames_per_clip += self.hparams.co3d_forward_frame_delay

        # From ActionRecognitionDatasets
        frames_per_clip = (
            1
            if self.hparams.co3d_forward_mode == "frame"
            else self.hparams.frames_per_clip
        )
        image_size = self.hparams.image_size
        dim_in = 3
        self.input_shape = (dim_in, frames_per_clip, image_size, image_size)

        CoX3D.__init__(
            self,
            dim_in,
            image_size,
            self.temporal_window_size,
            self.dataloader.num_classes,  # from ActionRecognitionDatasets
            self.hparams.x3d_conv1_dim,
            self.hparams.x3d_conv5_dim,
            self.hparams.x3d_num_groups,
            self.hparams.x3d_width_per_group,
            self.hparams.x3d_width_factor,
            self.hparams.x3d_depth_factor,
            self.hparams.x3d_bottleneck_factor,
            self.hparams.x3d_use_channelwise_3x3x3,
            self.hparams.x3d_dropout_rate,
            self.hparams.x3d_head_activation,
            self.hparams.x3d_head_batchnorm,
            self.hparams.x3d_fc_std_init,
            self.hparams.x3d_final_batchnorm_zero_init,
            self.hparams.co3d_temporal_fill,
        )

        receptive_field = (
            sum(
                [
                    m.kernel_size[0] - 1
                    for m in self.modules()
                    if "ConvCo3d" in str(type(m))
                ]
            )
            + self.temporal_window_size
        )
        logger.info(f"Model receptive field: {receptive_field} frames")

    def forward(self, x):
        if not hasattr(self, "_current_input_shape"):
            self._current_input_shape = x.shape

        if self._current_input_shape != x.shape:
            # Clean up state to accomodate new shape
            for m in self.modules():
                if hasattr(m, "clean_state"):
                    m.clean_state()

        result = None

        # Forward whole clip to init state
        if "clip" in self.hparams.co3d_forward_mode:
            result = CoX3D.forward3d(self, x[:, :, : self.temporal_window_size])

        # Flush state with intermediate frames
        if "init" in self.hparams.co3d_forward_mode:
            # Saturate by running example
            for i in range(1, self.hparams.co3d_forward_frame_delay + 1):
                CoX3D.forward(
                    self,
                    x[
                        :,
                        :,
                        (
                            self.temporal_window_size
                            if "clip" in self.hparams.co3d_forward_mode
                            else 0
                        )
                        + i,
                    ],
                )

        # Compute output for last frame(s)
        if "frame" in self.hparams.co3d_forward_mode:
            # Average over a number of frames
            result = CoX3D.forward(self, x[:, :, -self.hparams.co3d_num_forward_frames])
            for i in reversed(range(1, self.hparams.co3d_num_forward_frames)):
                result += CoX3D.forward(self, x[:, :, -i])
            result /= self.hparams.co3d_num_forward_frames

        return result

    def loss(self, preds, targets):
        print("Wazzzaaa?")
        return -1.0


if __name__ == "__main__":
    Main(CoX3DRide).argparse()
