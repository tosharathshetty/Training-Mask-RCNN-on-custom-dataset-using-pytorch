import os
import torch
from torchvision import transforms
import numpy as np
import cv2
import random
from tqdm import tqdm

# Randomly get some colors
def get_colors():
    colors = []
    for i in range(0,255,120):
        for j in range(0,255,120):
            for k in range(0,255,120):
                colors.append((i,j,k))
    colors.pop(0)
    random.shuffle(colors)
    return colors

# Colorize the mask
def colorize(mask,color):
    color_mask = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    color_mask[mask > 0] = color
    return color_mask

# Make prediction by mask-rcnn model and return every object's mask, box and class
def predict(img, model, class_name, threshold):
    img_tensor = transforms.ToTensor()(img).to(device,torch.float)
    result = model([img_tensor])
    scores = list(result[0]['scores'].cpu().detach().numpy())
    valid_num = sum(map(lambda x : x > threshold, scores))
    masks = (result[0]['masks']>0.5).squeeze().cpu().detach().numpy()[:valid_num]
    boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(result[0]['boxes'].detach().cpu().numpy().astype(int))][:valid_num]
    classes = [class_name[i] for i in list(result[0]['labels'].cpu().numpy())][:valid_num]
    return masks, boxes, classes

# Add masks, boxes and classes at objects in the image and represent them with different colors
def post_processing(img, colors, masks, boxes, classes):
    for i in range(len(masks)):
        color = colors[i%len(colors)]
        color_mask = colorize(masks[i],color)
        img = cv2.addWeighted(img, 1, color_mask, 0.5, 0)
        cv2.rectangle(img, boxes[i][0], boxes[i][1], color, 2)
        x,y = boxes[i][0]
        cv2.putText(img, classes[i], (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return img

# Input raw image and return final processed image
def get_result(img_path, model, class_name, threshold=0.5):
    img = cv2.imread(img_path)
    masks, boxes, classes = predict(img, model, class_name, threshold)
    result = post_processing(img, get_colors(), masks, boxes, classes)
    return result

if __name__=='__main__':
    # Load img's file names
    imgs = list(os.listdir(os.path.join('input_image')))
        
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"evaluating device is {device}")
    
    model = torch.load('person_classifier.pt')
    model.to(device)
    model.eval()
    
    # Define classes name
    class_name = ['__backgrounf__','person']
    
    # Process the input images and save them to the output folder
    for img in tqdm(imgs):
        result = get_result(img_path='input_image/'+img, model=model, class_name=class_name, threshold=0.5)
        cv2.imwrite('output_image/'+img.split('.')[0]+'.jpg', result)
    
    print('completed!')

