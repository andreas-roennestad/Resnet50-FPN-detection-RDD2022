
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models, transforms
from finetune import set_parameter_requires_grad, train_model, train
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from load_dataset import RoadCracksDetection
from dataloader import create_dataloaders
from torch.utils.data import Subset


print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
root_dir = "/cluster/projects/vc/courses/TDT17/2022/open/RDD2022/Norway/"
# Number of classes in the dataset
num_classes = 4

# Batch size for training (change depending on how much memory you have)
batch_size = 8

# Number of epochs to train for
num_epochs = 1

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True

model_ft =models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
print("Transforms: ", FasterRCNN_ResNet50_FPN_Weights.DEFAULT.transforms())

set_parameter_requires_grad(model_ft, feature_extract)
#print(model_ft)

#print(model_ft.roi_heads.box_head, model_ft.roi_heads.box_predictor)

num_ftrs = model_ft.roi_heads.box_predictor.bbox_pred.in_features
model_ft.roi_heads.box_predictor = FastRCNNPredictor(num_ftrs, num_classes)
input_size = 225 #MODIFY
data_transforms = FasterRCNN_ResNet50_FPN_Weights.DEFAULT.transforms()
# Data augmentation and normalization for training
# Just normalization for validation
"""data_transforms = {
    'train': transforms.Compose([
        #transforms.RandomResizedCrop(input_size),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Resize(input_size),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'target': transforms.Compose([
        #transforms.Resize(input_size),
        #transforms.CenterCrop(input_size),
        #transforms.ToTensor(),
        #transforms.Resize(input_size),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}"""

dataset = RoadCracksDetection(root_dir, "train", transforms=data_transforms)
dataset_test = RoadCracksDetection(root_dir, "test", transforms=data_transforms)

s_dataset = Subset(dataset, indices=range(len(dataset) // 400))
s_dataset_test = Subset(dataset, indices=range(len(dataset) // 400))
# Create training and validation dataloaders
dataloader = torch.utils.data.DataLoader(s_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=dataset.collate_fn)
dataloader_test = torch.utils.data.DataLoader(s_dataset_test, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=dataset.collate_fn)

"""dataloader, class_names = create_dataloaders(train_dir=root_dir,
                                                                    transform=data_transforms, # perform same data transforms on our own data as the pretrained model
                                                                    batch_size=32,
                                                                    num_workers=1) # set mini-batch size to 32"""


# Detect if we have a GPU available
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

# Send the model to GPU
model_ft = model_ft.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

loss_fn = nn.CrossEntropyLoss()


torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Start the timer
from timeit import default_timer as timer 
start_time = timer()

# Setup training and save the results
results = train(model=model_ft,
                       train_dataloader=dataloader,
                       test_dataloader=dataloader_test,
                       optimizer=optimizer_ft,
                       loss_fn=loss_fn,
                       epochs=5,
                       device=device)

# Train and evaluate
#model_ft, hist = train_model(model_ft, dataloader, loss_fn, optimizer_ft, num_epochs=num_epochs)

end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")
# End the timer and print out how long it took

