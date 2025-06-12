# Pneumonia Detection from Chest X-rays 🩻

[![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-orange)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/framework-PyTorch-red)](https://pytorch.org/)

A deep learning project for detecting pneumonia from chest X-ray images using a custom CNN in PyTorch, with an interactive Streamlit web app for predictions and model evaluation.

---

## Demo

![Streamlit App Screenshot](demo_screenshot.png) <!-- Add a screenshot or GIF of your app here -->

---

## Features

- Train a custom CNN for binary classification (Pneumonia vs. Normal)
- Data augmentation, class balancing, early stopping, and learning rate scheduling
- Model evaluation with confusion matrix and test predictions
- Streamlit web app for drag-and-drop predictions and model insights
- Downloadable sample images for easy testing

---

## Model Architecture

The model is a custom Convolutional Neural Network (CNN) inspired by VGG-style architectures, designed for binary classification of chest X-ray images (Pneumonia vs. Normal). The network is implemented in PyTorch and processes grayscale images of size 150x150.

### Layer-by-Layer Structure

- **Input:**  
  Grayscale images of shape `(1, 150, 150)`.

- **Convolutional Blocks:**  
  The network consists of 5 sequential convolutional blocks, each with:
  - A 2D convolutional layer (`Conv2d`) with a 3x3 kernel, stride 1, and padding 1.
  - Batch normalization (`BatchNorm2d`) for regularization and faster convergence.
  - ReLU activation.
  - 2x2 max pooling (`MaxPool2d`) for spatial downsampling.

  **Filter progression:**
  - Block 1: 8 filters
  - Block 2: 8 filters
  - Block 3: 8 filters
  - Block 4: 16 filters
  - Block 5: 16 filters

- **Flatten Layer:**  
  The output from the last convolutional block is flattened into a 1D vector.

- **Fully Connected (Dense) Layers:**  
  - Dense layer with 64 units + ReLU + Dropout (p=0.6)
  - Dense layer with 32 units + ReLU + Dropout (p=0.6)
  - Output layer: Dense with 1 unit + Sigmoid activation (for binary classification)

- **Regularization:**  
  - Dropout (p=0.6) after each dense layer to prevent overfitting.
  - L2 regularization (weight decay) in the optimizer.
  - Batch normalization after each convolutional layer.

- **Loss Function:**  
  - Binary Cross-Entropy Loss (`nn.BCELoss`), with class weights to handle class imbalance.

- **Optimizer:**  
  - Adam optimizer with a learning rate of 0.00015 and weight decay for L2 regularization.

- **Learning Rate Scheduling:**  
  - ReduceLROnPlateau: Reduces learning rate when validation loss plateaus.

- **Early Stopping:**  
  - Training stops early if validation loss does not improve for a set number of epochs (patience).

---

### Model Summary Table

| Layer Type         | Output Shape         | Parameters |
|--------------------|---------------------|------------|
| Input              | (1, 150, 150)       | 0          |
| Conv2d + BN + ReLU | (8, 150, 150)       | 80         |
| MaxPool2d          | (8, 75, 75)         | 0          |
| Conv2d + BN + ReLU | (8, 75, 75)         | 584        |
| MaxPool2d          | (8, 37, 37)         | 0          |
| Conv2d + BN + ReLU | (8, 37, 37)         | 584        |
| MaxPool2d          | (8, 18, 18)         | 0          |
| Conv2d + BN + ReLU | (16, 18, 18)        | 1,184      |
| MaxPool2d          | (16, 9, 9)          | 0          |
| Conv2d + BN + ReLU | (16, 9, 9)          | 2,336      |
| MaxPool2d          | (16, 4, 4)          | 0          |
| Flatten            | (256,)              | 0          |
| Dense (64) + ReLU  | (64,)               | 16,448     |
| Dropout (0.6)      | (64,)               | 0          |
| Dense (32) + ReLU  | (32,)               | 2,080      |
| Dropout (0.6)      | (32,)               | 0          |
| Dense (1) + Sigmoid| (1,)                | 33         |

*Note: Parameter counts are approximate and may vary depending on implementation details.*

---

### Design Rationale

- **VGG-style blocks** are simple and effective for image classification.
- **Batch normalization** and **dropout** are used to combat overfitting, which is common in medical imaging with limited data.
- **Class weights** and **data augmentation** help address class imbalance and improve generalization.
- **Early stopping** and **learning rate scheduling** ensure efficient and robust training.

---

## Project Structure

```
.
├── data/
│   └── chest_xray/
│       ├── train/
│       │   ├── NORMAL/
│       │   └── PNEUMONIA/
│       └── test/
│           ├── NORMAL/
│           └── PNEUMONIA/
├── src/
│   ├── models/
│   │   └── cnn_model.py
│   ├── app/
│   │   └── app.py
│   └── train.py
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd pneumonia-detection
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Training the Model

To train the model, use the following command:

```bash
python src/train.py --train_dir data/chest_xray/train --val_dir data/chest_xray/test --epochs 20 --batch_size 32
```

Optional arguments:
- `--epochs`: Number of training epochs (default: 20)
- `--batch_size`: Batch size for training (default: 32)
- `--model_save_path`: Path to save the trained model (default: models/pneumonia_model.pth)

## Running the Streamlit App

To run the Streamlit application:

```bash
streamlit run src/app/app.py
```

The application will open in your default web browser. You can then:
1. Upload a chest X-ray image
2. Click the "Detect Pneumonia" button
3. View the prediction results

## Data Augmentation

The training process includes data augmentation techniques:
- Random rotation
- Random horizontal flips
- Random affine transformations
- Image normalization

## Note

This model is for educational purposes only. Always consult healthcare professionals for medical diagnosis and treatment. 