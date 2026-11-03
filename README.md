# Sports Image Classification

A PyTorch image classification project using transfer learning to identify 100 different sports from images. The model uses a pretrained ResNet-50 backbone with a custom classification head, achieving **96.6% accuracy on a held-out test set**.

## Model Architecture

The model uses a pretrained ResNet-50 to extract image features, replacing its original ImageNet classification layer with a custom neural network head.

The classifier consists of:
- A 2048 → 512 fully connected layer
- ReLU activation and dropout (0.3)
- A final 512 → 100 output layer

Rather than retraining the entire network, the model freezes the earlier backbone layers and fine-tunes the final ResNet block (`layer4`) alongside the classifier. This allows the network to retain general visual features while adapting to the sports classification task.

## Training and Evaluation

The training pipeline uses:

- **Framework:** PyTorch and torchvision
- **Loss function:** Cross-entropy loss
- **Optimizer:** Adam (learning rate: 1e-4)
- **Batch size:** 32
- **Training epochs:** 4
- **Preprocessing:** ImageNet mean/std normalization

Training and validation losses are monitored after each epoch, along with validation accuracy. The final model is evaluated on a separate held-out test set.

**Reported test accuracy: 96.6%**

## Repository Structure

- `model.py` — ResNet-50 backbone and custom classifier architecture
- `train.py` — Data loading, training, validation, and test evaluation

## Running the Project

Install PyTorch and torchvision, then organize the dataset into `train/`, `valid/`, and `test/` directories using PyTorch's ImageFolder class-folder structure.

Run `python train.py` to train and evaluate the model.

The dataset and trained model weights are not included in this repository.
