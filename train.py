import torch 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import ResNetBackbone


# Image Normalization
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Preparing Datasets for Each Folder Type

train_dataset = datasets.ImageFolder(
    root="train",
    transform=train_transform
)

valid_dataset = datasets.ImageFolder(
    root="valid",
    transform=eval_transform
)

test_dataset = datasets.ImageFolder(
    root="test",
    transform=eval_transform
)

# DataLoaders

batch_size = 32  # or 64 if your GPU allows

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

images, labels = next(iter(train_loader))

#print(images.shape)  # [batch_size, 3, 224, 224]
#print(labels.shape)  # [batch_size]
#print(train_dataset.classes[:5])

model = ResNetBackbone()

features = model(images)
print(features.shape)