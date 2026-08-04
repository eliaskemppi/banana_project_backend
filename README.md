# Banana Ripometer: End-to-End Image Classification

A full-stack machine learning application that predicts the ripeness of bananas using Computer Vision. This project benchmarks a custom-built CNN against a fine-tuned MobileNetV2 to demonstrate the trade-offs between model complexity, accuracy, and inference speed.

## Data

### **Data aquisition**

I built my own custom dataset of pictures of bananas.

#### **Classes:** Under-ripe, Ripe, Spotty, Overripe.

Captured images under direct sunlight, kitchen LED lighting, low lighting etc. and varying backgrounds (countertops, wood, plates) to prevent the model from learning the background instead of the fruit. Also included multiple bananas of the same ripeness in photos.

#### **Split:**

Train/Val/Test. Due to the tediousness of collecting photos, the splits were uneven. The final evaluation was performed on the completely unseen test data.

#### **Data preprocessing and augmentation**

Raw images were transformed into tensors suitable for Deep Learning.

- Normalization: Applied ImageNet-standard normalization ($\mu=[0.485, 0.456, 0.406]$, $\sigma=[0.229, 0.224, 0.225]$) to align with MobileNetV2's pre-trained weights. 

- Geometry: Resized to $224 \times 224$ pixels.

**Augmentation:**

To artificially expand the dataset and reduce overfitting, I implemented:

- RandomHorizontalFlip (Bananas can face any direction).

- RandomRotation(20°) (Perspective shifts).

- ColorJitter (Simulating different camera sensor qualities).

## Model selection

## Future work

I could add a 'Not a banana' option for images that get low confidence for every banana ripeness.

Collecting more data for the dataset would likely improve model performance.

