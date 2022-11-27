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
from tqdm import tqdm
import csv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
predictions_file = "/cluster/work/andronn/VisualIntelligence/predictions.csv"

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def move_to(obj, device):
  if torch.is_tensor(obj):
    return obj.to(device)
  elif isinstance(obj, dict):
    res = {}
    for k, v in obj.items():
      res[k] = move_to(v, device)
    return res
  elif isinstance(obj, list):
    res = []
    for v in obj:
      res.append(move_to(v, device))
    return res
  else:
    raise TypeError("Invalid type for move_to")


"""
Contains functions for training and testing a PyTorch model.
"""

from typing import Dict, List, Tuple

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
        model: A PyTorch model to be trained.
        dataloader: A DataLoader instance for the model to be trained on.
        loss_fn: A PyTorch loss function to minimize.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of training loss and training accuracy metrics.
        In the form (train_loss, train_accuracy). For example:

        (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss =  0

    # Loop through data loader data batches
    for batch, (X, y, filename) in tqdm(enumerate(dataloader)):
        
        # Send data to target device
        #print(X[0])
        optimizer.zero_grad()
        X = move_to(X, device)
        y = move_to(y, device)
        with torch.set_grad_enabled(True):
            # 1. Forward pass
            #print(y)
            loss_dict = model(X, y)
            # 2. Calculate  and accumulate loss
            #loss = y_pred['loss_classifier']#loss_fn(y_pred, y)
            #train_loss += loss.item()
            loss = sum(loss for loss in loss_dict.values())
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        if batch % 30 == 0:
            print(f"Iteration #{batch} loss: {loss}")
            print(loss_dict)



    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    return train_loss

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device,
              write_csv=True) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
        model: A PyTorch model to be tested.
        dataloader: A DataLoader instance for the model to be tested on.
        loss_fn: A PyTorch loss function to calculate loss on the test data.
        device: A target device to compute on (e.g. "cuda" or "cpu").
        write_csv: must be run in single batches and no num_workers

    Returns:
        A tuple of testing loss and testing accuracy metrics.
        In the form (test_loss, test_accuracy). For example:

        (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval() 

    # Setup test loss and test accuracy values
    test_loss = 0

    # Turn on inference context manager
    with torch.no_grad():
        with torch.inference_mode():
            # Loop through DataLoader batches
            
            for batch, (X, y, f_name) in tqdm(enumerate(dataloader)):

            
                # Send data to target device
                #type(X)
                #print(f_name)
                X = move_to(X, device)
                #y = move_to(y, device)
        
                # 1. Forward pass
                # transport to cpu and save csvs
                predictions = model(X)
                print("Pred: ", predictions, '\n')
                print("y: ", y, '\n')
                for p in range(len(predictions)):
                    f = f_name[p]
                    boxes, labels, scores = predictions[p]['boxes'], predictions[p]['labels'], predictions[p]['scores']
                    line = ""
                    
                    for s in range(len(scores)):
                        if scores[s] > 0:     
                            b = boxes[s].cpu().numpy()
                            l = labels[s].cpu().numpy()
                            line += str(l) + ' '
                            for i in range(len(b)):
                                line+=str(int(b[i])) + ' '
                    with open(predictions_file, 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([f, line])
                    


    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    return test_loss

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          test_model=False,
          
          ) -> Dict[str, List]:
    """Trains and tests a PyTorch model.
    """
    # Create empty results dictionary
    results = {"train_loss": [],
        "test_loss": [],
    }
    test_loss = 0
    train_loss= 0
    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss = train_step(model=model,
                                            dataloader=train_dataloader,
                                            loss_fn=loss_fn,
                                            optimizer=optimizer,
                                            device=device
                                            )
        if test_model:
            test_loss = test_step(model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device)

        # Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"test_loss: {test_loss:.4f} | "
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)

    # Return the filled results at the end of the epochs
    return results


def test(model: torch.nn.Module, 
          test_dataloader: torch.utils.data.DataLoader, 
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
   
    # Create empty results dictionary
    results = {
        "test_loss": [],
    }

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        test_loss = test_step(model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device)

        # Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"test_loss: {test_loss:.4f} | "
        )

        # Update results dictionary
        results["test_loss"].append(test_loss)

    # Return the filled results at the end of the epochs
    return results

    