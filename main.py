## Code on kaggle https://www.kaggle.com/code/mistaluai/aifellowship-codingchallenge

from torch.utils.data import DataLoader
from dataset import FlowersDataset
from model import ResNet50
from tester import Evaluator
from trainer import Trainer
from utils.dataprocessor import DataProcessor
from utils.plotter import plot_test
from utils.random_seed import set_seed
import torch
import torch.optim as optim
from weighted_loss import WeightedCrossEntropyLoss
import torch.nn as nn
import torchvision.transforms.v2 as T

##Seed
seed = 2005
set_seed(seed=seed)

## Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

images_root = '/kaggle/working/data/images/jpg/'
labels_file = '/kaggle/working/data/imagelabels.mat'
splits_file = '/kaggle/working/data/setid.mat'

train_data, val_data, test_data = DataProcessor(images_root, labels_file, splits_file).get_data()


train_transforms = T.Compose([
    T.RandomResizedCrop(224, scale=(0.8, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandAugment(num_ops=2, magnitude=9),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])
train_dataset = FlowersDataset(train_data, transform=train_transforms)
val_dataset = FlowersDataset(val_data)
test_dataset = FlowersDataset(test_data)

batch_size = 250
num_workers = 4

train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)

dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}

fc_layer = [
    nn.Linear(2048, 1024),
    nn.BatchNorm1d(1024),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(1024, 103)
]
model = ResNet50(num_classes=103, fc_layer=fc_layer, freeze_backbone=False).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = WeightedCrossEntropyLoss(train_dataset, device, num_classes=103)

## Trainer
trainer = Trainer(model=model, optimizer=optimizer, loss_fn=criterion, dataloaders=dataloaders, device=device)

best_model, history = trainer.train(epochs=30, verbose=True)

## Evaluation
evaluator = Evaluator(best_model, test_loader, criterion, device)
avg_loss, metrics_dict, confusion_data = evaluator.evaluate(verbose=True)

## plotting
plot_test(avg_loss, metrics_dict, confusion_data)