# Transfer Learning Research Project

## Overview
This project investigates the performance of various transfer learning models trained on two datasets: the **Intel Image Dataset** and the **MIT Indoor Dataset**. The focus is on evaluating these models with different custom layers and comparing their performance across metrics such as accuracy, precision, recall, F1 score, specificity, and sensitivity.

## Models Used
1. **DenseNet169**
2. **EfficientNetB0**
3. **MobileNetV2**
4. **XceptionNet**
5. **ResNet50**

### Custom Layers
Each model was trained with three custom layer configurations:
1. **Global Average Pooling (GAP)**
2. **Average Depth Constraints**
3. **Depth Constraints**

## Datasets
1. **Intel Image Dataset**
   - 6 Classes
   - Image Size: 224x224
2. **MIT Indoor Dataset**
   - 67 Classes
   - Image Size: 512x512

### Data Augmentation Techniques
- Horizontal Flip
- Vertical Flip
- Zoom Range: 0.2
- Rotation Range: 360
- Width Shift: 0.1
- Height Shift: 0.1
- Channel Shift: 50
- Brightness Range: 0 to 1.2
- ImageNet Preprocessing

The target sizes for augmentation were adjusted based on the dataset being used:
- **224x224** for Intel Image Dataset
- **512x512** for MIT Indoor Dataset

### Dataset Generators
`ImageDataGenerator` was used for both training and validation generators.

## Methodology
1. **Model Initialization**
   - ImageNet weights were used for models trained on 512x512 images.
   - Models trained on 224x224 images did not use pretrained weights.

   Example Code:
   ```python
   keras.backend.clear_session()  # Clear backend
   shape = (512, 512, 3)
   input_tensor = keras.Input(shape=shape)
   base_model = keras.applications.<model_name>(input_tensor=input_tensor, weights='imagenet', include_top=False)
   ```

2. **Optimizer and Learning Rate Scheduler**
   - Optimizer: **SGD with momentum (0.9)**
   - Learning Rate Scheduler: **Exponential Decay**

3. **Metrics Tracked**
   - Accuracy
   - Loss
   - Precision
   - Recall
   - F1 Score
   - Specificity
   - Sensitivity

4. **Performance Evaluation**
   - Top-1 Accuracy
   - Top-5 Accuracy

## Results
The models achieved accuracies ranging from **69% to 92%** across datasets and configurations. Detailed results will include:
- Tables for top-1 and top-5 accuracy.
- Graphs for training and validation metrics across epochs.

## References
This research builds upon methodologies and findings presented in [this paper](https://arxiv.org/pdf/2104.12294).

## Contact
For any queries regarding this project, please reach out to:
**Kartik Garg**  
**Email:** gargkartik74@gmail.com  


