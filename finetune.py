from __future__ import print_function
from __future__ import division
import torch
from tqdm import tqdm
import csv
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Tuple

# Detect device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# File in which to save predictions
predictions_file = "/cluster/work/andronn/VisualIntelligence/predictions5.csv"
# Font for annotations
font_file = "/usr/share/fonts/liberation-mono/LiberationMono-Italic.ttf"



def move_to(obj, device):
    # Recursively move the elements of an object to device, used for annotations
    if torch.is_tensor(obj): 
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items(): res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj: res.append(move_to(v, device))
        return res
    else:
        raise TypeError("Invalid type for move_to")



def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps.

    Args:
        model: A PyTorch model to be trained.
        dataloader: A DataLoader instance for the model to be trained on.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        Value of train loss

    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss =  0

    # Loop through data loader data batches
    for batch, (X, y, filename) in tqdm(enumerate(dataloader)):
        
        # Send data to target device
        optimizer.zero_grad()

        X = move_to(X, device)
        y = move_to(y, device)
        with torch.set_grad_enabled(True):
            # Forward pass
            loss_dict = model(X, y)
            # Calculate  and accumulate loss
            loss = sum(loss for loss in loss_dict.values())
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        if batch % 200 == 0:
            print(f"Iteration #{batch} loss: {loss}")
            #print(loss_dict)



    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    return train_loss

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              device: torch.device,
              write_csv=True,
              draw_bbs=False) -> Tuple[float, float]:
    """Runs inference on test-set.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a dataset.

    Args:
        model: A PyTorch model to be tested.
        dataloader: A DataLoader instance for the model to be tested on.
        device: A target device to compute on (e.g. "cuda" or "cpu").
        write_csv: must be run in single batches and no num_workers.
        draw_bbs: boolean - visualize predictions and save or not

    Returns:
        Test running loss value
    """
    # Put model in eval mode
    model.eval() 

    # Setup test loss and test accuracy values
    test_loss = 0
    confidence = 0.4

    # Turn on inference context manager and specify nograd
    with torch.no_grad():
        with torch.inference_mode():

            # Loop through DataLoader batches
            for batch, (X, f_name) in tqdm(enumerate(dataloader)):

                # Send data to device
                X = move_to(X, device)
                #y = move_to(y, device)
        
                # Forward pass
                # transport to cpu and save csvs
                predictions = model(X)
                #print("Pred: ", predictions, '\n')
                #print("y: ", y, '\n')

                for p in range(len(predictions)):
                    f = f_name[p]
                    boxes, labels, scores = predictions[p]['boxes'], predictions[p]['labels'], predictions[p]['scores']
                    line = ""
                    for s in range(len(scores)):
                        #print(scores[s])
                        if scores[s] > 0.4:     
                            b = boxes[s].cpu().numpy()
                            l = labels[s].cpu().numpy()
                            line += str(l) + ' '
                            for i in range(len(b)):
                                line+=str(int(b[i])) + ' '
                    with open(predictions_file, 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([f, line])
                    if draw_bbs:
                        img = Image.open(r"/cluster/work/andronn/VisualIntelligence/Norway/test/images/{0}".format(f_name[p]))
                        font = ImageFont.truetype(font_file, size=40)
                        draw = ImageDraw.Draw(img)
                        for s in range(len(scores)):
                            if scores[s] > confidence: # Confidence th    
                                b = boxes[s].cpu().numpy()
                                l = labels[s].cpu().numpy() 
                                match l:
                                    # visualize pred
                                    case 1:
                                        draw.rectangle(b, outline="green",width=6)
                                        draw.text((b[0],b[3]), "D00", font=font)
                                    case 2:
                                        draw.rectangle(b, outline="red",width=6)
                                        draw.text((b[0],b[3]), "D10",  font=font)

                                    case 3:
                                        draw.rectangle(b, outline="orange",width=6)
                                        draw.text((b[0],b[3]), "D20",  font=font)
                                    case 4:
                                        draw.rectangle(b, outline="pink",width=6)
                                        draw.text((b[0],b[3]), "D40",  font=font)

                        # Save image with visualized predictions
                        img.save("/cluster/work/andronn/VisualIntelligence/predicted_images/{0}".format(f_name[p]))


                    

    #print("MAP: ", metric.compute())
    test_loss = test_loss / len(dataloader)
    return test_loss

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
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
                                            optimizer=optimizer,
                                            device=device,
                                            )
        if test_model:
            test_loss = test_step(model=model,
            dataloader=test_dataloader,
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
          epochs: int,
          device: torch.device) -> Dict[str, List]:
   
    # Create results dictionary
    results = {
        "test_loss": [],
    }


    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        test_loss = test_step(model=model,
            dataloader=test_dataloader,
            device=device,
            draw_bbs=False)

        # Print results
        print(
            f"Epoch: {epoch+1} | "
            f"test_loss: {test_loss:.4f} | "
        )

        # Update results dictionary
        results["test_loss"].append(test_loss)


    # Return the filled results at the end of the epochs
    return results

    
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False