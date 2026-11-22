# Chest X-ray Classification Using VGG16 and Explainable AI

## 1. Project Overview

This project focuses on the classification of chest X-ray images into three categories:

- **COVID-19**
- **Normal**
- **Pneumonia**

The main objective is to develop a deep learning model using **VGG16** and to explain its predictions using Explainable Artificial Intelligence (XAI) techniques.

Two explainability methods are used:

- **Grad-CAM:** highlights image regions that contribute to the model's prediction.
- **LIME:** identifies image regions that locally support or oppose a predicted class.

The project combines image classification, model evaluation, and visual explanations to better understand the behavior of a deep learning model on medical images.

This is an academic research project and not a medical diagnostic system.

---

## 2. Objectives

The main objectives are:

1. Prepare and verify a chest X-ray dataset.
2. Remove exact duplicate images.
3. Divide the dataset into training, validation, and testing sets.
4. Build a three-class classification model using transfer learning with VGG16.
5. Fine-tune the last convolutional block of VGG16.
6. Evaluate the model using classification metrics.
7. Explain predictions using Grad-CAM and LIME.
8. Compare the explanations provided by the two XAI methods.
9. Identify the limitations of the model and dataset.

---

## 3. Dataset

### Dataset source

The project uses the following public dataset:

**COVID19, Pneumonia and Normal Chest X-ray PA Dataset**

- Kaggle: https://www.kaggle.com/datasets/amanullahasraf/covid19-pneumonia-normal-chest-xray-pa-dataset
- Original dataset on Mendeley Data: https://data.mendeley.com/datasets/jctsfj2sfn/1

The dataset contains posterior-anterior (PA) chest X-ray images belonging to three classes: COVID-19, Normal, and Pneumonia.

Please refer to the original dataset page for its terms of use and citation requirements.

### Dataset preparation

The original files were checked for readability and exact duplicates.

MD5 hashing was used to identify identical files. Only one copy of each exact duplicate within a class was retained.

The final dataset contains **4,557 images**:

| Class | Number of images |
|---|---:|
| COVID-19 | 1,510 |
| Normal | 1,525 |
| Pneumonia | 1,522 |
| **Total** | **4,557** |

No exact duplicate images across different classes were detected during this integrity check.

### Dataset organization

The expected folder structure in Google Drive is:

```text
XAI_Research/
│
├── raw_kaggle_dataset/
│   └── COVID19_Pneumonia_Normal_Chest_Xray_PA_Dataset/
│       ├── covid/
│       ├── normal/
│       └── pneumonia/
│
├── models/
├── results/
│
├── expanded_dataset_manifest.csv
├── expanded_dataset_patient_audit.csv
└── expanded_dataset_split.csv
```

The dataset itself and trained model files are not included in this GitHub repository.

---

## 4. Data Integrity and Patient Grouping

### Duplicate detection

The MD5 hashing algorithm was used to identify exact duplicate images.

This step helps avoid counting identical files multiple times.

### Patient identification

Patient identifiers were extracted from filenames when available.

Two filename formats were considered:

- NIH format: `00000165_001.png`
- PERSON format: `person1000_bacteria_2931.jpeg`

Images with the same identified patient group are intended to remain in the same dataset partition.

However, COVID-19 images do not contain verifiable patient identifiers in their filenames. Therefore, their separation is performed at image level rather than patient level.

This means that complete patient-level independence cannot be guaranteed for the COVID-19 class.

---

## 5. Dataset Splitting

The dataset is divided into three subsets using `StratifiedGroupKFold`.

The objective is to maintain class proportions while keeping identified patient groups within a single partition.

The intended split is:

| Subset | Images | Percentage |
|---|---:|---:|
| Training | 3,253 | 71.38% |
| Validation | 648 | 14.22% |
| Testing | 656 | 14.40% |
| **Total** | **4,557** | **100%** |

- **Training set:** used to learn model parameters.
- **Validation set:** used to monitor training and select the best checkpoint.
- **Test set:** used for final model evaluation.

The test set is kept separate from model training.

Patient-level separation applies only where reliable group identifiers are available. For COVID-19 images, patient-level separation cannot be verified.

---

## 6. Image Preprocessing

All images are processed using TensorFlow.

The preprocessing pipeline includes:

1. Reading the image from its file path.
2. Decoding the image into three RGB channels.
3. Resizing it to 224 × 224 pixels.
4. Applying the VGG16 `preprocess_input` function.
5. Creating batches of 16 images.
6. Using prefetching to improve data loading performance.

### Preprocessing parameters

| Parameter | Value |
|---|---|
| Image size | 224 × 224 |
| Color channels | 3 (RGB) |
| Batch size | 16 |
| Preprocessing | VGG16 preprocess_input |
| Data pipeline | TensorFlow tf.data |

The same preprocessing is applied to the training, validation, and test datasets.

---

## 7. Model Architecture

### VGG16

VGG16 is a convolutional neural network originally trained on the ImageNet dataset.

In this project, transfer learning is used. The original classification layers are removed, and a new classification head is added for the three chest X-ray classes.

### Architecture

```text
Input Image (224 × 224 × 3)
          |
          v
    VGG16 Backbone
    (ImageNet weights)
          |
          v
 Global Average Pooling
          |
          v
    Dense (256)
      ReLU
          |
          v
   Dropout (0.5)
          |
          v
    Dense (3)
     Softmax
          |
          v
  COVID / Normal / Pneumonia
```

### Classification head

| Layer | Configuration |
|---|---|
| Backbone | VGG16, ImageNet pretrained |
| Global pooling | GlobalAveragePooling2D |
| Fully connected layer | Dense(256, ReLU) |
| Regularization | Dropout(0.5) |
| Output layer | Dense(3, Softmax) |

The Softmax layer returns a probability for each of the three classes.

The class with the highest probability is selected as the predicted class.

---

## 8. Training Strategy

Training is divided into two stages.

### Stage 1: Feature Extraction

During the first stage, all convolutional layers of VGG16 are frozen.

Only the newly added classification head is trained.

| Parameter | Value |
|---|---|
| Backbone | Frozen |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Maximum epochs | 15 |
| Loss function | Sparse Categorical Crossentropy |
| Metric | Accuracy |

This stage allows the classification head to learn how to distinguish the three classes using features already learned by VGG16.

### Stage 2: Fine-Tuning

During the second stage, the best Stage 1 checkpoint is loaded.

The last VGG16 block is unfrozen:

- block5_conv1
- block5_conv2
- block5_conv3
- block5_pool

The earlier convolutional blocks remain frozen.

| Parameter | Value |
|---|---|
| Trainable part | VGG16 block 5 and classification head |
| Optimizer | Adam |
| Learning rate | 0.00001 |
| Maximum epochs | 10 |
| Loss function | Sparse Categorical Crossentropy |
| Metric | Accuracy |

A lower learning rate is used to adapt the pretrained features without making large changes to the original weights.

### Training callbacks

The following callbacks are used:

- **ModelCheckpoint:** saves the best model according to validation loss.
- **EarlyStopping:** stops training when validation loss does not improve for four epochs and restores the best weights.
- **ReduceLROnPlateau:** reduces the learning rate when validation loss stops improving.
- **CSVLogger:** saves training and validation metrics in CSV files.

---

## 9. Model Evaluation

The final model is evaluated using the independent test dataset.

The evaluation includes:

### Accuracy

Measures the proportion of correctly classified images.

### Precision

Measures how many images predicted as a particular class actually belong to that class.

### Recall

Measures how many images belonging to a class are correctly detected.

### F1-score

Combines precision and recall into a single metric.

### Macro F1-score

Calculates the F1-score independently for each class and then takes their unweighted average.

This is useful because it gives equal importance to all three classes.

### Confusion Matrix

The confusion matrix shows the number of correct and incorrect predictions for each class.

It helps identify which classes are more frequently confused.

**Final evaluation results:** to be reported from the model retrained and evaluated using the corrected dataset split.

The final accuracy, Macro F1-score, classification report, and confusion matrix are generated in Section 8 of the notebook.

---

## 10. Explainable AI (XAI)

A classification model can provide accurate predictions without clearly explaining how it reached its decision.

For this reason, two explainability methods are used.

### 10.1 Grad-CAM

Grad-CAM stands for Gradient-weighted Class Activation Mapping.

It uses gradients of the selected class score with respect to convolutional feature maps to produce a heatmap.

In this project, the target convolutional layer is:

`block5_conv3`

The heatmap highlights image regions associated with the selected class score.

Three visualizations are generated:

1. Original chest X-ray.
2. Grad-CAM heatmap.
3. Original image with the heatmap overlay.

The heatmap helps inspect which regions are associated with the model's prediction.

### 10.2 LIME

LIME stands for Local Interpretable Model-agnostic Explanations.

It explains individual predictions by creating perturbed versions of an image and observing how the model's predictions change.

The image is divided into superpixels.

Some regions are hidden or modified, and the model is evaluated on the resulting images.

LIME then fits a simple local model to approximate the classifier's behavior around the selected image.

In this project:

| Parameter | Value |
|---|---|
| Number of perturbations | 1,000 |
| Number of displayed features | 10 |
| Segmentation unit | Superpixels |
| Explanation type | Local |

Positive weights indicate regions supporting the explained class.

Negative weights indicate regions opposing the explained class.

These weights are not probabilities and do not prove medical causality.

### 10.3 Comparison between Grad-CAM and LIME

| Grad-CAM | LIME |
|---|---|
| Gradient-based method | Model-agnostic method |
| Uses convolutional feature maps | Uses image perturbations |
| Produces a continuous heatmap | Produces superpixel-based regions |
| Shows regions associated with a class score | Shows local regions supporting or opposing a prediction |

Both methods provide complementary views of the model's behavior.

The same selected test examples are used for both methods to make the visual comparison more consistent.

---

## 11. Representative Examples

The notebook selects representative images from the test dataset:

- One correctly classified COVID-19 image.
- One correctly classified Normal image.
- One correctly classified Pneumonia image.
- One Pneumonia image incorrectly classified as Normal, if such an example exists.

These examples are analyzed using both Grad-CAM and LIME.

The objective is to inspect not only correct predictions but also possible classification errors.

---

## 12. Technologies and Libraries

The project uses the following technologies:

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| TensorFlow / Keras | Deep learning and model training |
| VGG16 | Pretrained CNN architecture |
| NumPy | Numerical operations |
| Pandas | Dataset and CSV management |
| Matplotlib | Visualization |
| Seaborn | Confusion matrix visualization |
| Scikit-learn | Dataset splitting and evaluation metrics |
| OpenCV | Image processing |
| Pillow | Image reading and verification |
| LIME | Local model explanations |
| Scikit-image | Superpixel visualization |
| Google Colab | Notebook execution |
| Google Drive | Dataset and model storage |
| GitHub | Source code versioning |

---

## 13. How to Run the Project

### Requirements

- A Google account.
- Google Colab.
- Access to the dataset.
- Google Drive storage.
- A GPU runtime is recommended for faster training.

### Step 1: Download the dataset

Download the dataset from:

https://www.kaggle.com/datasets/amanullahasraf/covid19-pneumonia-normal-chest-xray-pa-dataset

Extract it into the following Google Drive directory:

`MyDrive/XAI_Research/raw_kaggle_dataset/`

The three class folders must be named:

- `covid`
- `normal`
- `pneumonia`

### Step 2: Open the notebook

Open the notebook:

`XAI_VGG16_GRADCAM_LIME (1).ipynb`

Upload it to Google Colab or open it directly from Google Drive.

### Step 3: Connect Google Drive

Run the first notebook section to mount Google Drive.

The project directory is defined as:

`/content/drive/MyDrive/XAI_Research`

### Step 4: Prepare the dataset

Execute the data integrity and patient identification sections.

These steps create the dataset manifest and patient audit files.

### Step 5: Create the dataset split

Run the splitting section to generate the training, validation, and test partitions.

Check that all three classes are represented in every partition and that identified patient groups do not overlap.

### Step 6: Build the TensorFlow datasets

Run the preprocessing section to resize and preprocess the images.

### Step 7: Train the model

Execute Stage 1 and Stage 2.

The best checkpoints and training logs are saved in the `models` directory.

If checkpoints already exist, the corresponding saved model can be loaded instead of training again, provided it was trained using the same corrected split.

### Step 8: Evaluate the model

Load the best fine-tuned checkpoint and execute the test evaluation section.

### Step 9: Generate explanations

Execute the Grad-CAM and LIME sections.

The notebook generates visual explanations for the selected test examples.

---

## 14. Project Contributions

The main contributions of this project are:

1. **Dataset preparation:** verification of image readability and removal of exact duplicate files.
2. **Patient-group analysis:** extraction of patient identifiers from available filenames to reduce data leakage between partitions.
3. **Transfer learning:** implementation of VGG16 with ImageNet pretrained weights for three-class chest X-ray classification.
4. **Fine-tuning:** adaptation of the last VGG16 convolutional block to the target dataset.
5. **Model evaluation:** use of accuracy, precision, recall, F1-score, and confusion matrix.
6. **Explainability:** integration of Grad-CAM and LIME to inspect the regions associated with model predictions.
7. **Error analysis:** examination of representative correct predictions and a possible Pneumonia-to-Normal classification error.

The main contribution is the combination of classification and two complementary explainability methods in one workflow.

---

## 15. Limitations

Several limitations must be considered.

### Patient-level separation

Reliable patient identifiers are not available for the COVID-19 images.

Therefore, complete patient-level independence between training, validation, and test sets cannot be guaranteed for this class.

Some images from other classes may also lack identifiable patient information.

### Dataset heterogeneity

The images originate from different sources and may differ in image quality, acquisition conditions, formats, and characteristics.

The model may learn source-related patterns instead of only medically relevant features.

### Generalization

The evaluation is performed on a test subset of the available dataset.

The results do not guarantee the same performance on images from other hospitals, devices, or patient populations.

### Explainability limitations

Grad-CAM and LIME provide approximate explanations of model behavior.

They do not prove that the highlighted regions correspond to clinically meaningful findings.

### Clinical limitations

The model is an academic prototype.

It is not validated for clinical use and must not be used to diagnose patients or replace medical professionals.

---

## 16. Reference Study

This project is related to the following research article:

Ghnemat, R., Alodibat, S., and Abu Al-Haija, Q. (2023).

**Explainable Artificial Intelligence (XAI) for Deep Learning Based Medical Imaging Classification.**

Journal of Imaging, 9(9), 177.

https://doi.org/10.3390/jimaging9090177

The article discusses explainable AI methods for medical image classification and provides a scientific reference for this project.

The present implementation is an independent academic implementation using VGG16, Grad-CAM, and LIME. It should not be considered an exact reproduction of the article.

---

## 17. Repository Structure

```text
Chest-Xray-Classification-VGG16-XAI/
│
├── XAI_VGG16_GRADCAM_LIME (1).ipynb
├── README.md
└── .gitignore
```

The dataset, trained model checkpoints, and private credentials are excluded from the repository.

---

## 18. Conclusion

This project demonstrates how transfer learning with VGG16 can be applied to the classification of chest X-ray images into COVID-19, Normal, and Pneumonia categories.

It also demonstrates how Grad-CAM and LIME can be used to inspect model predictions through visual explanations.

The combination of classification, evaluation, and explainability provides a more informative view of model behavior than classification accuracy alone.

However, the limitations related to patient identifiers, dataset heterogeneity, generalization, and clinical validation must be considered when interpreting the results.
