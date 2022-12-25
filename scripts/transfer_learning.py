# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torchvision.models import resnet50, ResNet50_Weights
import matplotlib.pyplot as plt
import time
import os
import copy

import cv2
from tqdm import tqdm

# DATA_DIR = '/Users/flynn.wang/repo/flynn/thress_imgs/record_1124_target'
# CHECKPOINT_PATH = "/Users/flynn.wang/repo/flynn/thress_imgs/models/predict_num_v1129_v0.pt"

DATA_DIR = '/root/autodl-tmp/data/digits_1220/record_1220_target'
CHECKPOINT_PATH = "/root/autodl-tmp/data/digits_1220/record_1220_target/predict_num_v1224.pt"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"


def train_model(model,
                dataloaders,
                dataset_sizes,
                criterion,
                optimizer,
                scheduler,
                num_epochs=25):
  since = time.time()

  best_model_wts = copy.deepcopy(model.state_dict())
  best_acc = 0.0

  for epoch in range(num_epochs):
    print(f'Epoch {epoch}/{num_epochs - 1}')
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
      if phase == 'train':
        model.train()  # Set model to training mode
      else:
        model.eval()  # Set model to evaluate mode

      running_loss = 0.0
      running_corrects = 0

      # Iterate over data.
      for inputs, labels in tqdm(dataloaders[phase]):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
          outputs = model(inputs)
          _, preds = torch.max(outputs, 1)
          loss = criterion(outputs, labels)

          # backward + optimize only if in training phase
          if phase == 'train':
            loss.backward()
            optimizer.step()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
      if phase == 'train':
        scheduler.step()

      epoch_loss = running_loss / dataset_sizes[phase]
      epoch_acc = running_corrects.double() / dataset_sizes[phase]

      print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

      # deep copy the model
      if phase == 'val' and epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())

    print()

  time_elapsed = time.time() - since
  print(
      f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s'
  )
  print(f'Best val Acc: {best_acc:4f}')

  # load best model weights
  model.load_state_dict(best_model_wts)
  return model


def main():
  # Data augmentation and normalization for training
  # Just normalization for validation
  img_size = (224, 224)
  data_transforms = {
      'train':
      transforms.Compose([
          transforms.Resize(img_size),
          transforms.RandomResizedCrop(img_size,
                                       scale=(0.95, 1.05),
                                       ratio=(0.95, 1.05)),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
      'val':
      transforms.Compose([
          transforms.Resize(img_size),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
  }

  dataset_names = ['train', 'val']

  image_datasets = {
      x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
      for x in dataset_names
  }
  dataloaders = {
      x: torch.utils.data.DataLoader(image_datasets[x],
                                     batch_size=32,
                                     shuffle=True,
                                     num_workers=4)
      for x in dataset_names
  }
  dataset_sizes = {x: len(image_datasets[x]) for x in dataset_names}
  print(dataset_sizes)
  class_names = image_datasets['train'].classes

  print('number of classes: ', len(class_names))
  print(class_names)

  model_ft = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
  # for param in model_ft.parameters():
  #     param.requires_grad = False

  num_ftrs = model_ft.fc.in_features

  # Here the size of each output sample is set to 2.
  # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
  model_ft.fc = nn.Linear(num_ftrs, len(class_names))

  model_ft = model_ft.to(device)

  criterion = nn.CrossEntropyLoss()

  # Observe that all parameters are being optimized
  optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0005)

  # Decay LR by a factor of 0.1 every 7 epochs
  exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=1, gamma=0.1)
  model_ft = train_model(model_ft,
                         dataloaders,
                         dataset_sizes,
                         criterion,
                         optimizer_ft,
                         exp_lr_scheduler,
                         num_epochs=3)

  torch.save({
      "model_state_dict": model_ft.state_dict(),
  }, CHECKPOINT_PATH)


if __name__ == "__main__":
  main()
