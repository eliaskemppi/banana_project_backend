# Banana Ripometer: End-to-End Image Classification

#### https://banana-project-frontend.onrender.com/

Starting the app might take a minute because it's hosted for free

![Example](images/demo.png)

[![Watch a demo video](https://youtu.be/IBvfsqcrUNA/maxresdefault.jpg)](https://youtu.be/IBvfsqcrUNA)

A full-stack machine learning application that predicts the ripeness of bananas using Computer Vision. This project benchmarks a custom-built CNN against a fine-tuned MobileNetV2 to demonstrate the trade-offs between model complexity, accuracy, and inference speed. This is an end-to-end project which means I did all parts of the project myself (with the help of a bit of ChatGPT), including data acquisition, model training and evaluation, model selection and a Next.js frontend.

## Tech-Stack

Python, PyTorch, FastAPI, Docker, Next.js, TailwindCSS

## Data

### **Data acquisition**

I built my own custom dataset of self-collected pictures of bananas.

#### **Classes:** Under-ripe, Ripe, Spotty, Overripe.

Captured images under direct sunlight, kitchen LED lighting, low lighting etc. and varying backgrounds (countertops, wood, plates) to prevent the model from learning the background instead of the fruit. Also included multiple bananas of the same ripeness in photos.

#### **Split:**

Train/Val/Test. Due to the time required for data collection, the training set is class-imbalanced, with fewer overripe and spotty examples. Validation and test sets were balanced to provide a fair evaluation. The final evaluation was performed on the completely unseen test data.

Split sizes (188 images total):

- TRAIN: overripe: 17, ripe: 50, spotty: 22, underripe: 49

- VAL: overripe: 8, ripe: 8, spotty: 8, underripe: 10

- TEST: overripe: 8, ripe: 8, spotty: 8, underripe: 10

#### **Data preprocessing and augmentation**

Raw images were transformed into tensors suitable for Deep Learning.

- Normalization: Applied ImageNet-standard normalization ($\mu=[0.485, 0.456, 0.406]$, $\sigma=[0.229, 0.224, 0.225]$) to align with MobileNetV2's pre-trained weights. 

- Geometry: Resized to $224 \times 224$ pixels.

**Augmentation:**

To artificially expand the dataset and reduce overfitting, I implemented RandomHorizontalFlip to the training set.

## Model selection

I chose to display two models: A fine-tuned MobileNetV2 and a custom CNN built and trained from scratch. For both models, the checkpoint with the highest validation accuracy was selected to reduce overfitting.

## Model Evaluation

### MobileNetV2
Accuracy: 88.24%

Confusion Matrix
![MobileNetV2 Confusion Matrix](images/mobilenet_confusion.png)

### Custom CNN
Accuracy: 58.82%

Confusion Matrix
![CustomCNN Confusion Matrix](images/custom_confusion.png)

MobileNetV2 outperforms the custom CNN despite both being trained on the same dataset. This demonstrates the benefit of transfer learning on relatively small image datasets.

## Future work

- I could add a 'Not a banana' option for images that get low confidence for every banana ripeness.

- Collecting more data for the dataset would likely improve model performance.


