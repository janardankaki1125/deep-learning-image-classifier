# Deep Learning Image Classifier

Image classification using **PyTorch** with **Transfer Learning** (ResNet18 pretrained on ImageNet) on CIFAR-10 dataset.

## Goal
- Use pretrained model (Transfer Learning)
- Apply data augmentation
- Train the model and plot training curves
- Evaluate with classification metrics
- Save model for inference

## Features
- Pretrained ResNet18 with custom classifier head
- Data augmentation (RandomHorizontalFlip, RandomRotation)
- Training curves (Loss & Accuracy)
- Full classification report
- Model saved for inference

## Files
- `deep_learning_image_classifier.ipynb` → Main training notebook
- `requirements.txt` → Dependencies

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/janardankaki1125/deep-learning-image-classifier.git
   cd deep-learning-image-classifier
   pip install -r requirements.txt
   jupyter notebook
   model.load_state_dict(torch.load('cifar10_resnet18.pth', map_location=torch.device('cpu')))
