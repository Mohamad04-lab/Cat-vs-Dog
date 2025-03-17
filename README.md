# Cat vs Dog Classification using VGG16 and VGG19

## Overview
This project implements a deep learning-based **binary image classification** task to distinguish between cats and dogs using **VGG16** and **VGG19** convolutional neural networks (CNNs). The models are trained and evaluated on a labeled dataset to compare their performance in terms of accuracy, computational efficiency, and generalization.

## Models Used
1. **VGG16** - A 16-layer deep CNN known for efficient feature extraction.
2. **VGG19** - A deeper 19-layer variant of VGG16, designed for improved feature learning.

## Objectives
- Train and evaluate **VGG16 and VGG19** on a labeled dataset of cats and dogs.
- Compare their performance in terms of **accuracy, training time, and computational cost**.
- Implement **transfer learning** using pre-trained models from ImageNet.
- Provide visualizations of training progress and classification results.

## Installation
### Prerequisites
Ensure Python (>=3.8) is installed along with the necessary dependencies:

```sh
pip install tensorflow keras numpy pandas matplotlib scikit-learn opencv-python
```

## Dataset
The project uses the **Kaggle Dogs vs Cats dataset**, which consists of labeled images of cats and dogs.
- Download the dataset from [Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data).
- Extract and place it in the `data/` directory.

## Usage
To train and evaluate the models, run:

```sh
python train.py --model vgg16 --epochs 10 --batch_size 32
```

For VGG19:
```sh
python train.py --model vgg19 --epochs 10 --batch_size 32
```

### Example Execution
```sh
python train.py --model vgg16 --epochs 20 --batch_size 64 --output results/
```

## Output
- **Model Performance Metrics**: Training and validation accuracy/loss curves.
- **Classification Reports**: Precision, recall, and F1-score.
- **Visualization**: Sample predictions with class labels.

## Comparison Summary
| Model  | Accuracy | Training Time | Best Use Case |
|--------|----------|--------------|--------------|
| VGG16  | High     | Faster       | Quick feature extraction |
| VGG19  | Higher   | Slower       | More complex pattern recognition |

## Contribution Guidelines
We welcome contributions! To contribute:
1. Fork the repository.
2. Implement improvements (e.g., hyperparameter tuning, dataset augmentation).
3. Submit a pull request for review.
