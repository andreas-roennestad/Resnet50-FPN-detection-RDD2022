
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models, transforms
from finetune import set_parameter_requires_grad, train_model, train,test
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from load_dataset import RoadCracksDetection
from dataloader import create_dataloaders
from torch.utils.data import Subset
import pickle
import os

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
root_dir = "/cluster/projects/vc/courses/TDT17/2022/open/RDD2022/Norway/"
save_file = "/cluster/work/andronn/VisualIntelligence/resnet_fpn_model.pkl"

# Number of classes in the dataset
num_classes = 4

# Batch size for training (change depending on how much memory you have)
batch_size = 8

# Number of epochs to train for
num_epochs = 1

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True

if not os.path.isfile(save_file):
    with open(save_file,"r") as file:
        model_ft = torch.load(file)
else:
    print("Could not open file.\n")


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

data_transforms = FasterRCNN_ResNet50_FPN_Weights.DEFAULT.transforms()

dataset = RoadCracksDetection(root_dir, "train", transforms=data_transforms)
s_dataset_test = Subset(dataset, indices=range(len(dataset) // 20, len(dataset) // 20 + 200))
print("Length test data: ", len(s_dataset_test))


# Create validation dataloaders
dataloader_test = torch.utils.data.DataLoader(s_dataset_test, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=dataset.collate_fn)

print("Len dataloader test: ", len(dataloader_test))


# Send the model to GPU
model_ft = model_ft.to(device)


loss_fn = nn.CrossEntropyLoss()


torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Start the timer
from timeit import default_timer as timer 
start_time = timer()

# Setup training and save the results
results = test(model=model_ft,
                       test_dataloader=dataloader_test,
                       loss_fn=loss_fn,
                       epochs=num_epochs,
                       device=device)

# Train and evaluate
#model_ft, hist = train_model(model_ft, dataloader, loss_fn, optimizer_ft, num_epochs=num_epochs)

end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")
# End the timer and print out how long it took

torch.save(model_ft, save_file)
print("Model trained. Saved to {0}".format(save_file))