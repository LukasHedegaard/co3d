import os

from pytorch_lightning.utilities.parsing import AttributeDict
from ride.metrics import MetricDict, MetricMixin, OptimisationDirection
from torch import Tensor

from .ava_eval_helper import evaluate_ava, read_csv, read_exclusions, read_labelmap
from .ava_helper import load_image_lists


class AvaMetric(MetricMixin):

    # def validate_attributes(self):
    #     for attribute in ["hparams.loss", "classes"]:
    #         attrgetter(attribute)(self)

    def __init__(self, hparams: AttributeDict, *args, **kwargs):
        self.full_ava_test = False  # cfg.AVA.FULL_TEST_ON_VAL
        self.mode = "test"  # TODO: Remove

        data_dir = "/mnt/archive/common/datasets/ava/data"  # TODO: Fix
        annotation_dir = "/mnt/archive/common/datasets/ava/splits"  # TODO: Fix
        self.excluded_keys = read_exclusions(
            os.path.join(annotation_dir, "ava_val_excluded_timestamps_v2.2.csv")
        )
        self.categories, self.class_whitelist = read_labelmap(
            os.path.join(
                annotation_dir,
                "ava_action_list_v2.2_for_activitynet_2019.pbtxt",
            )
        )
        gt_filename = os.path.join(annotation_dir, "ava_val_v2.2.csv")
        self.full_groundtruth = read_csv(gt_filename, self.class_whitelist)

        _, self.video_idx_to_name = load_image_lists(
            frame_list_dir=annotation_dir + "/frame_lists",
            train_lists=["train.csv"],
            test_lists=["val.csv"],
            frame_dir=data_dir + "/frames",
            is_train=(self.mode == "train"),
        )

    @classmethod
    def _metrics(cls):
        return {"ava": OptimisationDirection.MAX}

    def metrics_epoch(
        self, preds: Tensor, targets: Tensor, *args, **kwargs
    ) -> MetricDict:
        result = evaluate_ava(
            preds,
            targets["ori_boxes"],
            targets["metadata"],
            self.excluded_keys,
            self.class_whitelist,
            self.categories,
            groundtruth=self.full_groundtruth,
            video_idx_to_name=self.video_idx_to_name,
        )

        return {"ava_map": float(result)}
