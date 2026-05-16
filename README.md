# EEG Emotion Classification using Deep Learning

## Overview

This project presents a complete deep learning pipeline for **emotion classification using EEG (Electroencephalogram) signals**. The notebook preprocesses high-dimensional EEG data and compares the performance of three neural network architectures:

* **1D Convolutional Neural Network (CNN)**
* **Artificial Neural Network (ANN)**
* **Long Short-Term Memory Network (LSTM)**

The objective is to classify human emotional states from EEG signal features into three categories:

* **POSITIVE**
* **NEGATIVE**
* **NEUTRAL**

The project demonstrates how different deep learning architectures interpret EEG feature representations and evaluates their effectiveness for emotion recognition tasks.

---

# Dataset Information

The dataset contains extracted EEG numerical features and corresponding emotion labels.

### Dataset Shape

* **Rows (Samples):** 2132
* **Columns:** 2549

  * 2548 EEG feature columns
  * 1 label column

### Example Features

* Mean values
* Differential mean values
* FFT (Fast Fourier Transform) features
* Frequency-domain EEG characteristics

### Target Labels

* POSITIVE
* NEGATIVE
* NEUTRAL

---

# Project Workflow

## 1. Data Preprocessing

Several preprocessing steps are applied before training:

### Feature & Label Separation

* EEG features are separated from emotion labels.

### Label Encoding

Text labels are converted into numerical form using `LabelEncoder`.

Example:

* NEGATIVE → 0
* NEUTRAL → 1
* POSITIVE → 2

### Feature Normalization

`StandardScaler` is used to standardize the data:

* Mean = 0
* Standard Deviation = 1

This improves model convergence and training stability.

### Data Reshaping

For CNN and LSTM architectures, the data is reshaped into 3D format:

```python
(samples, timesteps, features)
```

Resulting shape:

```python
(2132, 2548, 1)
```

### Train-Test Split

The dataset is divided into:

* **80% Training**
* **20% Testing**

using:

```python
train_test_split(test_size=0.2, random_state=42)
```

---

# Deep Learning Models

## 1D CNN Model

### Purpose

The CNN captures **local spatial patterns** and neighboring feature relationships within EEG signals.

### Architecture

* Conv1D Layer (64 filters)
* Batch Normalization
* MaxPooling1D
* Conv1D Layer (128 filters)
* Batch Normalization
* MaxPooling1D
* Flatten Layer
* Dense Layers
* Dropout Regularization
* Softmax Output Layer

### Advantages

* Excellent feature extraction capability
* Learns local EEG signal structures
* High classification accuracy

### Final Test Accuracy

# 🎯 98.59%

---

## ANN Model

### Purpose

The ANN learns **global relationships** among EEG features through fully connected layers.

### Architecture

* Dense(512)
* Batch Normalization
* Dropout
* Dense(256)
* Dense(128)
* Softmax Output Layer

### Advantages

* Simpler architecture
* Faster training
* Strong performance on tabular EEG features

### Final Test Accuracy

# 🎯 97.66%

---

## LSTM Model

### Purpose

LSTM networks are designed to capture **temporal dependencies** and sequential information in EEG data.

### Architecture

* LSTM(128)
* Batch Normalization
* Dropout
* LSTM(64)
* Dense(64)
* Softmax Output Layer

### Advantages

* Handles sequential dependencies
* Suitable for time-series EEG analysis

### Final Test Accuracy

# 🎯 91.57%

---

# Model Performance Comparison

| Model | Test Accuracy |
| ----- | ------------- |
| CNN   | 98.59%        |
| ANN   | 97.66%        |
| LSTM  | 91.57%        |

---

# Key Findings

## CNN Achieved the Best Performance

The CNN model produced the highest accuracy because:

* EEG features contain strong local spatial patterns
* Convolutional layers effectively extract discriminative signal characteristics
* Pooling layers reduce noise and improve generalization

## ANN Also Performed Extremely Well

The ANN demonstrated that:

* EEG features are highly informative even without convolution operations
* Fully connected layers can successfully classify emotions from extracted EEG statistics

## LSTM Performed Relatively Lower

Possible reasons:

* The dataset consists of extracted features rather than raw temporal EEG sequences
* Temporal dependencies may already be partially lost during feature extraction
* LSTMs generally perform better on raw sequential signal data

---

# Technologies Used

## Programming Language

* Python

## Libraries

* NumPy
* Pandas
* Scikit-learn
* TensorFlow / Keras

---

# Installation

Install required libraries:

```bash
pip install numpy pandas scikit-learn tensorflow
```

---

# How to Run

## 1. Clone the Repository

```bash
git clone <repository-link>
```

## 2. Open the Notebook

Open the Jupyter Notebook or Google Colab file.

## 3. Upload Dataset

Place the dataset file:

```bash
Emotion_Classification_EEG_Data_.csv
```

inside your working directory or Google Drive.

## 4. Run All Cells

Execute all notebook cells sequentially.

---

# Sample Training Output

## CNN Results

```python
Test Accuracy: 0.9859
```

## ANN Results

```python
Test Accuracy: 0.9766
```

## LSTM Results

```python
Test Accuracy: 0.9157
```

---

# Future Improvements

Possible future enhancements include:

* Using raw EEG signals instead of extracted features
* Applying Transformer-based architectures
* Performing hyperparameter optimization
* Using attention mechanisms
* Applying cross-validation
* Testing on larger EEG datasets
* Implementing real-time emotion recognition systems

---

# Applications

EEG-based emotion recognition can be applied in:

* Brain-Computer Interfaces (BCI)
* Mental health monitoring
* Human-computer interaction
* Adaptive gaming systems
* Emotion-aware AI systems
* Medical diagnosis support systems

---

# Conclusion

This project successfully demonstrates the effectiveness of deep learning techniques for EEG-based emotion classification.

Among the tested architectures:

* **CNN achieved the best performance (98.59%)**
* ANN also achieved highly competitive results
* LSTM showed moderate performance due to the nature of extracted EEG features

The project highlights the importance of selecting architectures that align with the structural characteristics of EEG data.

---

# Author

**Shayan**

Email: [shayanrokhva1999@gmail.com](mailto:shayanrokhva1999@gmail.com)
