import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

import cv2

from torchvision.models import resnet50, ResNet50_Weights

NUM_CLASSES = 11
CLASS_NAMES = ['0', '1', '12', '192', '2', '24', '3', '384', '48', '6', '96']


def create_digit_model(checkpoint_path: str,
                       num_classes: int = NUM_CLASSES,
                       device=torch.device('cpu')):
  model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
  model.eval()

  num_ftrs = model.fc.in_features

  # Adding a new layer
  model.fc = nn.Linear(num_ftrs, num_classes)

  checkpoint_state = torch.load(checkpoint_path, map_location=device)
  model.load_state_dict(checkpoint_state['model_state_dict'])
  model = model.to(device)
  return model


preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def predict_digit(model, img, class_names=CLASS_NAMES):
  import torchvision.transforms.functional as transform
  img = transform.to_pil_image(img)

  batch = preprocess(img).unsqueeze(0)

  # Step 4: Use the model and print the predicted category
  prediction = model(batch).squeeze(0).softmax(0)

  class_id = prediction.argmax().item()
  score = prediction[class_id].item()
  category_name = class_names[class_id]

  return category_name, score
