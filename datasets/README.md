# Dataset Preparation
Below, you find instructions for retrieving the individual datasets.
Once downloaded, 
The path to the datasets should be specified in `.env`:
```bash
DATASETS_PATH=/path/to/datasets
```

It is assumed that datasets follow the structure
```
dataset_name
|_ data
|  |_ [video name 0]
|  |  |_ [video name 0]_000001.jpg
|  |  |_ [video name 0]_000002.jpg
|  |  |_ ...
|  |_ [video name 1]
|     |_ [video name 1]_000001.jpg
|     |_ [video name 1]_000002.jpg
|     |_ ...
|_ splits
|  |_ train.csv
|  |_ val.csv
```

## Kinetics-400
[Kinetics](https://deepmind.com/research/open-source/kinetics) is a large-scale dataset for Trimmed Human Activity Recognition, consisting of 10 second videos collected from YouTube, ranging over 400 classes.
Due to it's origin, a direct download of the complete dataset is not possible.
Instead, a list of videos and corresponding labels can be downloaded [here](https://storage.googleapis.com/deepmind-media/Datasets/kinetics400.tar.gz), and a YouTube Crawler can subsequently be employed to collect the videos one by one. Note: this process may take multiple days.

It is assumed that the dataset has the following structure:
```
kinetics400
|_ data
|  |_ validate
|  |  |_ abseiling/
|  |  |_ air drumming/
|  |  |  |_ 3caPS4FHFF8_000036_000046.mp4*
|  |  |  |_ 3yaoNwz99xM_000062_000072.mp4*
|  |  |  |_ ...
|  |  |_ ...
|  |_ test
|  |  |_ ...
|  |_ train
|  |  |_ ...
|_ splits
|  |_ test.csv*
|  |_ test.json*
|  |_ train.csv*
|  |_ train.json*
|  |_ validate.csv*
|  |_ validate.json*
```


## Charades
[Charades](https://prior.allenai.org/projects/charades) is multi-label action classification dataset comprised of 9848 videos of daily indoors activities collected through Amazon Mechanical Turk.
Steps to prepare dataset.

1. Please download the Charades RGB frames from the [dataset provider](http://ai2-website.s3.amazonaws.com/data/Charades_v1_rgb.tar).

2. Download the *frame list* from the following links: ([train](https://dl.fbaipublicfiles.com/pyslowfast/dataset/charades/frame_lists/train.csv), [val](https://dl.fbaipublicfiles.com/pyslowfast/dataset/charades/frame_lists/val.csv)).

It is assumed that the dataset has the following structure:
```
charades
|_ data
|  |_ Charades_v1_rgb
|  |  |_ 001YG/
|  |  |  |_ 001YG-000001.jpg
|  |  |  |_ 001YG-000002.jpg
|  |  |  |_ ...
|  |  |_ 003WS/
|  |  |  |_ ...
|  |  |_ ...
|_ splits
|  |_ train.csv
|  |_ val.csv
|  |_ Charades_v1_test.csv
|  |_ Charades_v1_train.csv
|  |_ ...
