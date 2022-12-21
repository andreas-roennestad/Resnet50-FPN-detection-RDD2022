
import torch
import torch.optim as optim
import torchvision
from torchvision import models
from finetune import train
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from load_dataset import RoadCracksDetection
from torch.utils.data import Subset


print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
root_dir = "/cluster/work/andronn/VisualIntelligence/Norway/"
save_file = "/cluster/work/andronn/VisualIntelligence/resnet_fpn_model_3.pkl"

# Number of classes in the dataset
num_classes = 5


# Batch size for training (change depending on how much memory you have)
batch_size = 2

# Number of epochs to train for
num_epochs = 38

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True

# Initialize pretrained model with weights frozen for untrainable layers
model_ft = models.detection.fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT, trainable_backbone_layers=3)

print("Transforms: ", FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT.transforms())

num_ftrs = model_ft.roi_heads.box_predictor.bbox_pred.in_features
model_ft.roi_heads.box_predictor = FastRCNNPredictor(num_ftrs, num_classes)
print(model_ft)

data_transforms = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT.transforms()
# Data augmentation and normalization for training
# Just normalization for validation


dataset = RoadCracksDetection(root_dir, 'train', transforms=data_transforms)
s_dataset = Subset(dataset, indices=range(0, len(dataset)))
print("Length training data: ", len(s_dataset))


# Create dataloaders
dataloader = torch.utils.data.DataLoader(s_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=dataset.collate_fn)

print("Len dataloader training: ", len(dataloader))

# Detect if we have a GPU available
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

# Send the model to GPU
model_ft.to(device)

# Gather the parameters to be optimized/updated in this run
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

# Using SGD as optimizer
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9, weight_decay=0.0005)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Start the timer
from timeit import default_timer as timer 
start_time = timer()

# Train
results = train(model=model_ft,
                       train_dataloader=dataloader,
                       test_dataloader=None,
                       optimizer=optimizer_ft,
                       epochs=num_epochs,
                       device=device, 
                       test_model=False,
                       )


end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")
# End the timer and print out how long it took

torch.save(model_ft, save_file)
#torch.save(model_ft.state_dict(), save_file)
print("Model trained. Saved to {0}".format(save_file))
