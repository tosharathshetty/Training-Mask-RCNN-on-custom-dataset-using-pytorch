import os,sys
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import math
from tqdm import tqdm

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, image_set='images_train', annotation_set='annotations_train'):
        self.root = root
        self.image_set = image_set
        self.annotation_set = annotation_set
        
        # Load all image and annotation files and make sure they are sorted
        self.imgs = list(sorted(os.listdir(os.path.join(root, image_set))))
        self.anns = list(sorted(os.listdir(os.path.join(root, annotation_set))))

    def __getitem__(self, idx):
        
        # Load images and annotations
        img_path = os.path.join(self.root, self.image_set, self.imgs[idx])
        ann_path = os.path.join(self.root, self.annotation_set, self.anns[idx])
        img = Image.open(img_path)
        ann = np.load(ann_path, allow_pickle='TRUE').item()
        
        # Convert annotation into torch Tensor
        boxes = torch.as_tensor(ann['boxes'], dtype=torch.float32)
        labels = torch.as_tensor(ann['labels'], dtype=torch.int64)
        masks = torch.as_tensor(ann['masks'], dtype=torch.uint8)
        
        # Convert image into torch Tensor
        convert_tensor = transforms.ToTensor()
        img = convert_tensor(img)
        
        # Put annotations into 'target' which torchvision.models.detection.maskrcnn_resnet50_fpn required
        target = {}    
        target['boxes'] = boxes
        target['labels'] = labels
        target['masks'] = masks
        
        # Return required inputs of model which are input tensor and target
        return img, target

    def __len__(self):
        return len(self.imgs)

# Define dataloader's collate function
def my_collate_fn(batch):
    return tuple(zip(*batch))

def build_model(num_classes):
    # Load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights='DEFAULT')

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # Replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    lr_scheduler = None
    
    # Build a dictionary to recored losses
    LOSS = {
        'loss_classifier':[],
        'loss_box_reg':[],
        'loss_mask':[],
        'loss_objectness':[],
        'loss_rpn_box_reg':[],
        'loss_sum':[]
    }
    
    # Warm up the model in the first epoch
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )
        
    # Training process
    for images, targets in tqdm(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        for i in loss_dict.keys():
            LOSS[i].append(loss_dict[i].item())    
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        LOSS['loss_sum'].append(loss_value)

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

    return LOSS

def val_one_epoch(model, data_loader, device, epoch):
    # Build a dictionary to recored losses
    LOSS = {
        'val_loss_classifier':[],
        'val_loss_box_reg':[],
        'val_loss_mask':[],
        'val_loss_objectness':[],
        'val_loss_rpn_box_reg':[],
        'val_loss_sum':[]
    }
    
    # Speed ​​up evaluation by not computing gradients
    with torch.no_grad():
        model.train()
        
        # Evaluating process
        for images, targets in tqdm(data_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            for i in loss_dict.keys():
                LOSS["val_"+i].append(loss_dict[i].item())    
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            LOSS['val_loss_sum'].append(loss_value)

    return LOSS
