# Training Mask R-CNN on custom dataset using pytorch
This repository contains code for training a Mask R-CNN model on a custom dataset using PyTorch. Mask R-CNN is a powerful deep learning model that can be used for both object detection and instance segmentation. This project serves as a practical demonstration of how to train a Mask R-CNN model on a custom dataset using PyTorch, with a focus on building a person classifier.

One way to save time and resources when building a Mask RCNN model is to use a pre-trained model. In this repository, we will use a pre-trained model with a ResNet-50-FPN backbone provided by torchvision. This model has already undergone extensive training on the COCO dataset, allowing it to learn generalizable features from a large dataset. Fine-tuning this pre-trained model to suit a specific task can significantly reduce the amount of time and resources required compared to training a new model from scratch. In this case, We only need to modify the box predictor layer and the mask predictor layer to fit the custom dataset.

![4](https://user-images.githubusercontent.com/48171500/230904606-5b72102b-121c-4b54-a4e7-5aea19064417.jpg)


# Installation
To run this project, you will need to install the following packages:

* PyTorch
* torchvision
* NumPy
* matplotlib
* Pillow
* OpenCV
* tqdm
* wget
* pycocotools
* pycocotools-windows (for windows user)
* requests

You can install these packages using pip:
```
pip install torch torchvision numpy matplotlib Pillow opencv-python tqdm wget pycocotools requests
```

# Dataset
Before you can start training the model, you will need to prepare your custom dataset. The dataset should be organized in the following directory structure:
```
dataset/
    images_train/
        data1.jpg
        data2.jpg
        data3.jpg
        ...
    annotations_train/
        data1.npy
        data2.npy
        data3.npy
        ...
```

Each image should have a corresponding NPY file with the same name, containing the annotations for that image. The NPY files should contain the following information for each object in the image:

* **label** -> **int** : object class
* **box** -> **np.array(shape=(4, ))** : bounding box coordinates (xmin, ymin, xmax, ymax)
* **mask** -> **np.array(shape=(width, height))** : segmentation mask

The final NPY file should be structured as follows:
```py
{
    'boxes': np.array([box_1, box_2, ...]), 
    'labels': np.array([label_1, label_2, ...]),  
    'masks': np.array([mask_1, mask_2, ...])
}
```
where each element of the boxes, labels, and masks arrays corresponds to a single object in the image, and the length of each array is the number of objects in the image.

In this repository, I have chosen to work with the person data from the COCO dataset. To utilize the person images and annotations from the COCO dataset, we need to download the entire dataset first. Once downloaded, we should extract the images and annotations for persons and organize them into the structure mentioned above. The specific steps for doing so can be found in the [data_prepare.ipynb](data_prepare.ipynb) notebook. However, if you are working with a different custom dataset, the specific steps may vary depending on the structure and format of your dataset.

# Usage
## Training
To train your own custom dataset, you should first clone this repository.
```
git clone https://github.com/duck00036/Training-Mask-RCNN-on-custom-dataset-using-pytorch.git
```
Delete the .gitkeep file in **dataset/images_train** and **dataset/annotations_train**, then put your own images and annotations into these two folder.

Run the script:
```
python train.py
```
and the training will start with default parameters.

Or your can use [training_demo.ipynb](training_demo.ipynb) notebook to train your model step by step.

### default parameters

* **Class Number** : 1 + 1(background)
* **Batch Size** : 1
* **Optimizer** : SGD (momentum = 0.9, weight_decay = 0.0005)
* **Learning Rate** : 0.0001
* **lr scheduler** : No
* **Epoch Number** : 20
* **Saving Frequency** : every 5 epoch

Of course, you can tweak it as much as you want by editing the code to make the model fit your data better.

## Person classifier
To use the trained person classifier, you should first clone this repository.
```
git clone https://github.com/duck00036/Training-Mask-RCNN-on-custom-dataset-using-pytorch.git
```
Download the trained model [here](https://duck00036-public-images.s3.ap-northeast-1.amazonaws.com/person_classifier.pt) in the repository, and put the images to be classified into the **input_image** folder

Run the script:
```
python eval.py
```
and the result will be saved in the **output_image** folder.

# About trained person classifier model
At the beginning, I attempted to train the model using a small dataset. However, due to the complexity of the ResNet50 model, it was too large compared to my sample size, leading to overfitting after a few initial epochs. To address this issue, I gradually lowered the learning rate, but the validation loss continued to be unsatisfactory. As a result, I made the decision to increase the sample size to 20,000 and experimented with various optimizers and hyperparameters such as learning rate, momentum, and weight decay. Ultimately, after several rounds of experimentation, I arrived at a set of optimal parameters that generated the desired results.

[**final_model**](https://duck00036-public-images.s3.ap-northeast-1.amazonaws.com/person_classifier.pt)
### fianl parameters

* **Class Number** : 1 + 1(background)
* **Batch Size** : 4
* **Optimizer** : SGD (momentum = 0.9, weight_decay = 0.0005)
* **Learning Rate** : 0.00002
* **lr scheduler** : No
* **Epoch Number** : 33 with lowest valid loss

### losses curve
![losses_curve](https://user-images.githubusercontent.com/48171500/230793409-ec36f082-6fbc-4f6f-8622-d0286fae4c92.png)
After the 33rd epoch, the valid loss starts to increase, so I choose to stop here and take the model with the lowest valid loss.

I also evaluated the box mPA and segmentation mPA of the model with the evaluation tool provided by the COCO dataset, and the results looks not bad:

**box mAP**

 |       Indicator        |       IoU            | area   |        maxDets         | Value|
 |       :--:             |       :--:           | :--:   |        :--:            | :--: |
 |Average Precision  (AP) | 0.50:0.95            |   all  | 100                    | 0.569|
 |Average Precision  (AP) | 0.50                 |  all   | 100                    | 0.825|
 |Average Precision  (AP) | 0.75                 |   all  | 100                    | 0.620|
 |Average Precision  (AP) | 0.50:0.95            | small  | 100                    | 0.320|
 |Average Precision  (AP) | 0.50:0.95            | medium | 100                    | 0.574|
 |Average Precision  (AP) | 0.50:0.95            | large  | 100                    | 0.729|
 |Average Recall     (AR) | 0.50:0.95            |   all  | 1                      | 0.195|
 |Average Recall     (AR) | 0.50:0.95            |   all  | 10                     | 0.584|
 |Average Recall     (AR) | 0.50:0.95            |  all   | 100                    | 0.675|
 |Average Recall     (AR) | 0.50:0.95            | small  | 100                    | 0.514|
 |Average Recall     (AR) | 0.50:0.95            | medium | 100                    | 0.696|
 |Average Recall     (AR) | 0.50:0.95            | large  | 100                    | 0.793|

**segmentation mAP**

 |       Indicator        |       IoU            | area   |        maxDets         | Value|
 |       :--:             |       :--:           | :--:   |        :--:            | :--: |
 |Average Precision  (AP) | 0.50:0.95            |   all  | 100                    | 0.476|
 |Average Precision  (AP) | 0.50                 |  all   | 100                    | 0.795|
 |Average Precision  (AP) | 0.75                 |   all  | 100                    | 0.517|
 |Average Precision  (AP) | 0.50:0.95            | small  | 100                    | 0.224|
 |Average Precision  (AP) | 0.50:0.95            | medium | 100                    | 0.471|
 |Average Precision  (AP) | 0.50:0.95            | large  | 100                    | 0.634|
 |Average Recall     (AR) | 0.50:0.95            |   all  | 1                      | 0.171|
 |Average Recall     (AR) | 0.50:0.95            |   all  | 10                     | 0.505|
 |Average Recall     (AR) | 0.50:0.95            |  all   | 100                    | 0.575|
 |Average Recall     (AR) | 0.50:0.95            | small  | 100                    | 0.419|
 |Average Recall     (AR) | 0.50:0.95            | medium | 100                    | 0.599|
 |Average Recall     (AR) | 0.50:0.95            | large  | 100                    | 0.687|
 
During the evaluation process, I also found something that might be a bit strange, that is, sometimes the prediction results of the model will be more accurate than the annotations of the coco dataset, or it can be said that the annotations of the coco dataset is not always so accurate.
![ex1](https://user-images.githubusercontent.com/48171500/230794761-fde8d4de-6561-48f6-87d5-e2025303ef9b.jpg)
![ex2](https://user-images.githubusercontent.com/48171500/230794764-ce2a4641-35fb-4cda-8555-333f63979e41.jpg)
![ex3](https://user-images.githubusercontent.com/48171500/230794770-d5314b41-e5cf-44d8-95d1-159e1e8b8420.jpg)
This is not to say that coco dataset is not good, it is still a very good dataset in general, but if the purpose of the model is to identify specific types of objects, a well-labeled custom dataset may further improve the performance of the model, which is one of my future work.

# Demo
## image
![1](https://user-images.githubusercontent.com/48171500/230904654-9a08ed9f-8945-4830-a32f-4d10d1b132ad.jpg)
![2](https://user-images.githubusercontent.com/48171500/230904662-10b6bff7-c5ca-44c0-a2b4-60a4d29a2346.jpg)
![6](https://user-images.githubusercontent.com/48171500/230907553-1c726968-20c5-4a87-af8b-578256e036e8.jpg)
## video
![v3](https://user-images.githubusercontent.com/48171500/230919202-3048c152-6759-4644-b298-1b81566fafab.gif)

# Application
Using the segmentation function of this model, I combined it with another GAN model that can cartoonize images and realized the function of cartooning real people in photos and videos.

More information can be found in [this repository](https://github.com/duck00036/Cartoonize-people-in-image)

![a2f](https://user-images.githubusercontent.com/48171500/230923691-17cf6677-70cd-432d-adfc-28c8a9e0a999.jpg)
![a6f](https://user-images.githubusercontent.com/48171500/230923715-c18dc5cc-5861-44ad-84cf-1fcf6fd98081.jpg)
![a_v2f](https://user-images.githubusercontent.com/48171500/230933181-17a3b910-805d-42fb-b3e9-c6a7aa73857b.gif)

# reference
[TORCHVISION OBJECT DETECTION FINETUNING TUTORIAL](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html#torchvision-object-detection-finetuning-tutorial)

[TORCHVISION MASKRCNN_RESNET50_FPN DOCUMENTATION](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn.html#torchvision.models.detection.maskrcnn_resnet50_fpn)


