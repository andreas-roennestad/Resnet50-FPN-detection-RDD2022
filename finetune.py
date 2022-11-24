from __future__ import print_function
from __future__ import division
from load_dataset import RoadCracksDetection
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
root_dir = "/cluster/projects/vc/courses/TDT17/2022/open/RDD2022/Norway/"


dataset = RoadCracksDetection(root_dir)

item = dataset.__getitem__(0)
print(item[0])
print(item[1])
