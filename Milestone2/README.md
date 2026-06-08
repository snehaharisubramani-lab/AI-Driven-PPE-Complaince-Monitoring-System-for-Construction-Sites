# Milestone 2 - PPE Dataset Annotation & Model Training

## What We Did
- Collected 4 datasets from Roboflow Universe
  (Helmet, Vest, Goggles, Boots)
- Merged all 4 into one unified project: PPE-Compliance-Final
- Cleaned 46 duplicate classes into 11 standardized classes
- Applied preprocessing: Auto-Orient + Resize 640x640
- Applied augmentations: Flip, Rotation, Brightness, Blur, Noise
- Generated 29,366 images (3x augmentation)
- Started YOLOv11 Nano model training on Google Colab T4 GPU

## Dataset Details
- Total Images: 11,942
- Augmented: 29,366 images
- Classes: 11
- Model: YOLOv11 Nano
- Training Platform: Google Colab (Tesla T4 GPU)

## Classes
- helmet / no_helmet
- vest / no_vest
- goggles / no_goggles
- boots / no_boots
- gloves / no_gloves
- person

## Tools Used
- Roboflow (dataset preparation)
- Google Colab (model training)
- YOLOv11 Nano (model architecture)
