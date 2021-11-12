# Modified from https://github.com/facebookresearch/SlowFast/blob/master/slowfast/datasets/ava_helper.py

import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

FPS = 30
AVA_VALID_FRAMES = range(902, 1799)


def load_image_lists(frame_list_dir, train_lists, test_lists, frame_dir, is_train):
    """
    Loading image paths from corresponding files.

    Args:
        cfg (CfgNode): config.
        is_train (bool): if it is training dataset or not.

    Returns:
        image_paths (list[list]): a list of items. Each item (also a list)
            corresponds to one video and contains the paths of images for
            this video.
        video_idx_to_name (list): a list which stores video names.
    """
    logger.debug("Loading image lists")
    list_filenames = [
        os.path.join(frame_list_dir, filename)
        for filename in (train_lists if is_train else test_lists)
    ]
    image_paths = []
    video_idx_to_name = []
    for list_filename in list_filenames:
        df = pd.read_csv(list_filename, sep=" ")
        # df["path"] = df["path"].apply(partial(os.path.join,frame_dir))
        df["path"] = f"{frame_dir}/" + df["path"].astype(str)  # much faster
        df_groups = df.groupby("video_id")
        image_paths.extend(df_groups["path"].apply(list).tolist())
        video_idx_to_name.extend(
            df_groups["original_vido_id"].apply(lambda x: next(iter(set(x)))).tolist()
        )

    return image_paths, video_idx_to_name


def load_boxes_and_labels(
    train_gt_box_lists,
    train_predict_box_lists,
    test_predict_box_list,
    annotation_dir,
    detection_score_thresh,
    full_test_on_eval,
    mode,
):
    """
    Loading boxes and labels from csv files.

    Args:
        cfg (CfgNode): config.
        mode (str): 'train', 'val', or 'test' mode.
    Returns:
        all_boxes (dict): a dict which maps from `video_name` and
            `frame_sec` to a list of `box`. Each `box` is a
            [`box_coord`, `box_labels`] where `box_coord` is the
            coordinates of box and 'box_labels` are the corresponding
            labels for the box.
    """
    logger.debug("Loading boxes and labels")
    gt_lists = train_gt_box_lists if mode == "train" else []
    pred_lists = train_predict_box_lists if mode == "train" else test_predict_box_list
    ann_filenames = [
        os.path.join(annotation_dir, filename) for filename in gt_lists + pred_lists
    ]
    ann_is_gt_box = [True] * len(gt_lists) + [False] * len(pred_lists)

    # Only select frame_sec % 4 = 0 samples for validation if not
    # set FULL_TEST_ON_VAL.
    boxes_sample_rate = 4 if mode == "val" and not full_test_on_eval else 1
    all_boxes, count, unique_box_count = parse_bboxes_file(
        ann_filenames=ann_filenames,
        ann_is_gt_box=ann_is_gt_box,
        detect_thresh=detection_score_thresh,
        boxes_sample_rate=boxes_sample_rate,
    )
    # logger.info("Finished loading annotations from: %s" % ", ".join(ann_filenames))
    # logger.info("Detection threshold: {}".format(detection_score_thresh))
    # logger.info("Number of unique boxes: %d" % unique_box_count)
    # logger.info("Number of annotations: %d" % count)

    return all_boxes


def get_keyframe_data(boxes_and_labels):
    """
    Getting keyframe indices, boxes and labels in the dataset.

    Args:
        boxes_and_labels (list[dict]): a list which maps from video_idx to a dict.
            Each dict `frame_sec` to a list of boxes and corresponding labels.

    Returns:
        keyframe_indices (list): a list of indices of the keyframes.
        keyframe_boxes_and_labels (list[list[list]]): a list of list which maps from
            video_idx and sec_idx to a list of boxes and corresponding labels.
    """

    def sec_to_frame(sec):
        """
        Convert time index (in second) to frame index.
        0: 900
        30: 901
        """
        return (sec - 900) * FPS

    keyframe_indices = []
    keyframe_boxes_and_labels = []
    count = 0
    for video_idx in range(len(boxes_and_labels)):
        sec_idx = 0
        keyframe_boxes_and_labels.append([])
        for sec in boxes_and_labels[video_idx].keys():
            if sec not in AVA_VALID_FRAMES:
                continue

            if len(boxes_and_labels[video_idx][sec]) > 0:
                keyframe_indices.append((video_idx, sec_idx, sec, sec_to_frame(sec)))
                keyframe_boxes_and_labels[video_idx].append(
                    boxes_and_labels[video_idx][sec]
                )
                sec_idx += 1
                count += 1

    return keyframe_indices, keyframe_boxes_and_labels


def get_num_boxes_used(keyframe_indices, keyframe_boxes_and_labels):
    """
    Get total number of used boxes.

    Args:
        keyframe_indices (list): a list of indices of the keyframes.
        keyframe_boxes_and_labels (list[list[list]]): a list of list which maps from
            video_idx and sec_idx to a list of boxes and corresponding labels.

    Returns:
        count (int): total number of used boxes.
    """

    tot = 0
    maximum = 0
    for video_idx, sec_idx, _, _ in keyframe_indices:
        count = len(keyframe_boxes_and_labels[video_idx][sec_idx])
        tot += count
        if count > maximum:
            maximum = count
    return tot, maximum


def parse_bboxes_file(ann_filenames, ann_is_gt_box, detect_thresh, boxes_sample_rate=1):
    """
    Parse AVA bounding boxes files.
    Args:
        ann_filenames (list of str(s)): a list of AVA bounding boxes annotation files.
        ann_is_gt_box (list of bools): a list of boolean to indicate whether the corresponding
            ann_file is ground-truth. `ann_is_gt_box[i]` correspond to `ann_filenames[i]`.
        detect_thresh (float): threshold for accepting predicted boxes, range [0, 1].
        boxes_sample_rate (int): sample rate for test bounding boxes. Get 1 every `boxes_sample_rate`.
    """
    all_boxes = {}
    count = 0
    unique_box_count = 0
    for filename, is_gt_box in zip(ann_filenames, ann_is_gt_box):
        with open(filename, "r") as f:
            for line in f:
                row = line.strip().split(",")
                # When we use predicted boxes to train/eval, we need to
                # ignore the boxes whose scores are below the threshold.
                if not is_gt_box:
                    score = float(row[7])
                    if score < detect_thresh:
                        continue

                video_name, frame_sec = row[0], int(row[1])
                if frame_sec % boxes_sample_rate != 0:
                    continue

                # Box with format [x1, y1, x2, y2] with a range of [0, 1] as float.
                box_key = ",".join(row[2:6])
                box = list(map(float, row[2:6]))
                label = -1 if row[6] == "" else int(row[6])

                if video_name not in all_boxes:
                    all_boxes[video_name] = {}
                    for sec in AVA_VALID_FRAMES:
                        all_boxes[video_name][sec] = {}

                if box_key not in all_boxes[video_name][frame_sec]:
                    all_boxes[video_name][frame_sec][box_key] = [box, []]
                    unique_box_count += 1

                all_boxes[video_name][frame_sec][box_key][1].append(label)
                if label != -1:
                    count += 1

    for video_name in all_boxes.keys():
        for frame_sec in all_boxes[video_name].keys():
            # Save in format of a list of [box_i, box_i_labels].
            all_boxes[video_name][frame_sec] = list(
                all_boxes[video_name][frame_sec].values()
            )

    return all_boxes, count, unique_box_count
