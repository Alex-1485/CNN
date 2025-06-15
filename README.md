# Malaria Detection using a Custom Convolutional Neural Network

This project was developed for the **Pearson BTEC Level 4 Higher National Certificate in Digital Technologies**, as part of the unit **Unit 15: Fundamentals of AI and Intelligent Systems**.

## Project Overview

This project implements a system for the automated detection of malaria in microscopic blood cell images. The goal is to classify images as either "Parasitized" or "Uninfected" using a deep learning model.

To demonstrate a foundational understanding of deep learning principles, a **custom Convolutional Neural Network (CNN) architecture was designed and trained from scratch** using the TensorFlow and Keras frameworks. This approach provides full control over the model's design and showcases the process of building an effective image classifier without relying on pre-trained models.

## Dataset

The model was trained on the **Malaria dataset** from TensorFlow Datasets.

- **Source:** [https://www.tensorflow.org/datasets/catalog/malaria](https://www.tensorflow.org/datasets/catalog/malaria)
- **Description:** The dataset contains 27,558 cell images, with a balanced distribution of parasitized and uninfected cells.
- **Classes:**
  - `Parasitized`: A blood cell infected with the malaria parasite.
  - `Uninfected`: A healthy blood cell.

## Methodology

The project follows a standard machine learning workflow:

### 1. Data Preparation
- **Loading:** The dataset was loaded directly using the `tensorflow_datasets` library.
- **Data Splitting:** The data was partitioned into training (70%), validation (15%), and testing (15%) sets.
- **Preprocessing:** All images were resized to `128x128` pixels and pixel values were normalized to the `[0, 1]` range.
- **Augmentation:** The training dataset was augmented with random flips, rotations, and zooms to improve model generalization and prevent overfitting.

### 2. Model Architecture
A custom CNN was designed with the following structure:

1.  **Input Layer:** Expects images of shape `(128, 128, 3)`.

2.  **Three Convolutional Blocks:** The core of the feature extractor consists of three sequential blocks. Each block is designed to learn progressively more complex features:
    - **Block 1:** `Conv2D` with 32 filters -> `BatchNormalization` -> `MaxPooling2D`.
    - **Block 2:** `Conv2D` with 64 filters -> `BatchNormalization` -> `MaxPooling2D`.
    - **Block 3:** `Conv2D` with 128 filters -> `BatchNormalization` -> `MaxPooling2D`.
    *Batch Normalization* is used after each convolution to stabilize and accelerate training.

3.  **Flatten Layer:** Converts the 2D feature maps from the convolutional blocks into a 1D vector.

4.  **Classifier Head:**
    - A `Dense` layer with 128 neurons and ReLU activation.
    - A `Dropout` layer with a rate of 0.5 for strong regularization to combat overfitting.
    - The final `Dense` output layer with 1 neuron and a `sigmoid` activation function, which outputs a probability score for the binary classification.

### 3. Model Training
- **Framework:** TensorFlow with the Keras API.
- **Compiler Settings:**
  - **Optimizer:** `Adam` (learning rate = 0.001)
  - **Loss Function:** `BinaryCrossentropy`
  - **Metrics:** `accuracy`, `precision`, `recall`
- **Training:** The model was trained from scratch for 25 epochs to allow the weights to converge effectively.

## Results

The model's performance was evaluated on the unseen test set.



- **Test Accuracy:** `[Insert your accuracy here, e.g., 0.8815]`
- **Test Precision:** `[Insert your precision here, e.g., 0.8750]`
- **Test Recall:** `[Insert your recall here, e.g., 0.8890]`

### Training History
The training and validation curves demonstrate a stable learning process.

### Confusion Matrix
The confusion matrix provides a clear breakdown of the model's performance across both classes.

## How to Run the Project

1.  **Environment:** The project is best run in a Google Colab environment with a GPU runtime selected (`Runtime` -> `Change runtime type` -> `GPU`).
2.  **Dependencies:** The notebook uses standard Python libraries, primarily `tensorflow`, `numpy`, `matplotlib`, and `seaborn`.
3.  **Execution:** Open the `.ipynb` file in Google Colab and run all cells sequentially from top to bottom. The dataset will be downloaded automatically by `tensorflow_datasets`.
