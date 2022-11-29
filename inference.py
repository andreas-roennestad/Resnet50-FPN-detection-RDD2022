
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models, transforms
from finetune import set_parameter_requires_grad, train,test
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
save_file = "/cluster/work/andronn/VisualIntelligence/resnet_fpn_model_1.pkl"

# Number of classes in the dataset
num_classes = 5

# Batch size for training (change depending on how much memory you have)
batch_size = 6

# Number of epochs to train for
num_epochs = 1

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params

"""if not os.path.isfile(save_file):
    print("Could not open file. \n")
    exit()
else:
    with open(save_file,"rb") as file:
        model_ft =models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        num_ftrs = model_ft.roi_heads.box_predictor.bbox_pred.in_features
        model_ft.roi_heads.box_predictor = FastRCNNPredictor(num_ftrs, num_classes)
        model_ft.load_state_dict(torch.load(save_file, map_location='cpu'), strict=True)"""
model_ft = torch.load(save_file)
print("Loaded model from ", save_file)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

data_transforms = FasterRCNN_ResNet50_FPN_Weights.DEFAULT.transforms()

dataset_test = RoadCracksDetection(root_dir, "test", transforms=data_transforms)
#s_dataset_test = Subset(dataset, indices=range(len(dataset)//10*8, len(dataset)))
print("Length test data: ", len(dataset_test))


# Create validation dataloaders
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0, collate_fn=dataset_test.collate_fn)
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


end_time = timer()
print(f"[INFO] Total time: {end_time-start_time:.3f} seconds")
# End the timer and print out how long it took
