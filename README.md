# Representative Forgery Mining for Fake Face Detection

This repository is a modified version of the [Representative Forgery Mining](https://github.com/chengruiwang/RFM) implementation, adapted for thesis requirements. The original implementation was created by [Chengrui Wang](https://github.com/chengruiwang).

## About

This project implements a Representative Forgery Mining (RFM) approach for fake face detection, featuring advanced feature masking techniques and improved generalization capabilities. The implementation is based on the original work by Wang and Deng (CVPR 2021), with modifications and enhancements for thesis research.

## Features

- Representative Forgery Mining (RFM) for improved model generalization
- Feature Activation Map (FAM) visualization
- Support for multiple datasets (FaceForensics++, Celeb-DF, DFFD)
- Custom dataset training capabilities
- Distributed training support
- Comprehensive evaluation metrics
- Flexible configuration system

## Requirements

- Python 3.8+
- PyTorch 1.7.0+
- CUDA-compatible GPU
- OpenCV
- NumPy
- Other dependencies listed in requirements.txt

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/RFM.git
cd RFM
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

- `train.py`: Main training script
- `train_custom.py`: Custom dataset training script
- `test_model.py`: Model evaluation script
- `AvgFAM.py`: Feature activation map visualization
- `utils/`: Utility functions and dataset classes
  - `utils.py`: Common utilities
  - `custom_dataset.py`: Custom dataset implementation
  - `datasets_profiles.py`: Dataset configurations
  - `DataTools.py`: Data loading utilities
- `models/`: Saved model checkpoints
- `logs/`: Training logs

## Usage

### Training

1. Standard Training:

```bash
python train.py
```

2. Custom Dataset Training:

```bash
python train_custom.py --dataset_path /path/to/your/dataset
```

### Testing

To evaluate a trained model:

```bash
python test_model.py --model_path /path/to/model --dataset_path /path/to/dataset
```

### Visualization

Generate average Feature Activation Maps (FAM):

```bash
python AvgFAM.py
```

## Model Architecture

The RFM model consists of:

- Xception backbone network
- Feature masking mechanism
- Classification head
- FAM generation for visualization

## Original Repository

This project is based on the original implementation by Wang and Deng:

- Original Paper: [Representative Forgery Mining for Fake Face Detection](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Representative_Forgery_Mining_for_Fake_Face_Detection_CVPR_2021_paper.html)
- Original Authors: Chengrui Wang and Weihong Deng

## Thesis Modifications

This version includes modifications and enhancements made for thesis research purposes, including:

- Custom dataset handling
- Modified training pipeline
- Enhanced evaluation metrics
- Additional configuration options
- Improved error handling
- Extended logging capabilities

## Acknowledgments

- Original implementation by Wang and Deng (CVPR 2021)
- All contributors to the original repository
- Thesis advisors and research team
