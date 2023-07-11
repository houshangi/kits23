# KiTS23

The official repository of the 2023 Kidney Tumor Segmentation Challenge

[Challenge Homepage](https://kits23.kits-challenge.org/)

**Current Dataset Version: `0.1.2`**

## Timeline

- **April 14**: Training dataset release
- **July 14**: Deadline for short paper
- **July 21 - July 28**: Prediction submissions accepted
- **July 31**: Results announced
- **October 8 or 12**: Satellite event at MICCAI 2023

## News

Check here for the latest news about the KiTS23 dataset and starter code!

## Usage

This repository is meant to serve two purposes:

1. To help you **download the dataset**
2. To allow you to benchmark your model using the **official implementation of the metrics**

We recommend starting by installing this `kits23` package using pip. Once you've done this, you'll be able to use the command-line download entrypoint and call the metrics functions from your own codebase.

### Installation

This should be as simple as cloning and installing with pip.

```bash
git clone https://github.com/neheller/kits23
cd kits23
pip3 install -e .
```

We're running Python 3.10.6 on Ubuntu and we suggest that you do too. If you're running a different version of Python 3 and you discover a problem, please [submit an issue](https://github.com/neheller/kits23/issues/new) and we'll try to help. Python 2 or earlier is not supported. If you're running Windows or MacOS, we will do our best to help but we have limited ability to support these environments.

### Data Download

Once the `kits23` package is installed, you should be able to run the following command from the terminal.

```bash
kits23_download_data
```

This will place the data in the `dataset/` folder.

### Using the Metrics

We provide a reference implementation for the metrics and the ranking scheme so that participants know exactly how their algorithm is going to be validated. We strongly encourage using these implementations for model selection during development. In order to do this, we recommend training all your models as cross-validations, so that you have predictions for the entire training set with which you can meaningfully evaluate your approaches.

#### Compute metrics for your predictions

After installing the KiTS23 repository, the following console command is available to you: `kits23_compute_metrics`. You can use it to evaluate predictions against the training ground truth. It's as simple as

```bash
kits23_compute_metrics FOLDER_WITH_PREDICTIONS -num_processes XX
```

This will produce a `evaluation.csv` in FOLDER_WITH_PREDICTIONS with the computed Dice and surface Dice scores.

#### Ranking code

Ranking is based on first averaging your dice and surface dice scores across all cases and HECs, resulting in two values: your average Dice and average Surface Dice. We then use 'rank-then-aggregate' to merge these metrics into a final ranking.

Here are the steps you need to do to run the ranking locally (for example, to find the best configuration for your submission):

1) Execute `generate_summary_csv`, located in [ranking.py](ranking.py). The documentation will tell you how
2) Then use `rank_participants` from the same file with the summary.csv generated by `generate_summary_csv` to generate the final ranking.

## License and Attribution

The code in this repository is under an MIT License. The data that this code downloads is under a CC BY-NC-SA (Attribution-NonCommercial-ShareAlike) license. If you would like to use this data for commercial purposes, please contact Nicholas Heller at helle246@umn.edu. Please note, we do not consider training a model for the *sole purpose of participation in this competition* to be commercial use. Therefore, industrial teams are strongly encouraged to participate. If you are an academic researcher interested in using this dataset in your work, you needn't ask for permission.

If this project is useful to your research, please cite our most recent KiTS challenge paper in Medical Image Analysis \[[html](https://www.sciencedirect.com/science/article/abs/pii/S1361841520301857)\] \[[pdf](https://arxiv.org/pdf/1912.01054.pdf)\].

```bibtex
@article{heller2020state,
  title={The state of the art in kidney and kidney tumor segmentation in contrast-enhanced CT imaging: Results of the KiTS19 Challenge},
  author={Heller, Nicholas and Isensee, Fabian and Maier-Hein, Klaus H and Hou, Xiaoshuai and Xie, Chunmei and Li, Fengyi and Nan, Yang and Mu, Guangrui and Lin, Zhiyong and Han, Miofei and others},
  journal={Medical Image Analysis},
  pages={101821},
  year={2020},
  publisher={Elsevier}
}
```