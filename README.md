# DIO Project Challenge

## Image Classification with Transfer Learning

---

##  Challenge Description

This project was developed as part of a **Digital Innovation One (DIO)** challenge, aiming to apply and reinforce the concepts of **Transfer Learning** in Deep Learning using **Python** in the **Google Colab** environment.

The challenge consists of training an image classification model for flower categories and comparing two approaches:

* A Convolutional Neural Network (CNN) trained **from scratch**
* A model trained using **Transfer Learning** with a pre-trained network

---

##  Project Objective

To apply the **Transfer Learning** technique using a pre-trained deep neural network and evaluate its performance compared to a model trained from scratch for image classification.

---

##  Technologies and Tools

* Python
* Google Colab
* TensorFlow / Keras
* NumPy
* Pandas
* Matplotlib

---

##  Dataset

* **Source:** Kaggle
* **Dataset:** Flowers Dataset
* **Link:** [https://www.kaggle.com/datasets/l3llff/flowers](https://www.kaggle.com/datasets/l3llff/flowers)

The original dataset contains **16 flower categories**. For this challenge, only **2 classes** were selected, while the remaining categories were excluded to simplify the classification task and focus on the Transfer Learning technique.

---

##  Project Steps

### 1️ Data Loading and Preprocessing

* Dataset download using `kagglehub`
* Image resizing to **224×224** pixels
* Pixel normalization
* Label encoding using **one-hot encoding**
* Dataset split:

  * 70% Training
  * 15% Validation
  * 15% Test

---

### 2️ Training a Model from Scratch

* Construction of a custom **Convolutional Neural Network (CNN)**
* Use of `Conv2D`, `MaxPooling`, `Dropout`, and `Dense` layers
* Activation functions: `ReLU` and `Softmax`
* Model compilation using **Adam optimizer** and **Categorical Cross-Entropy loss**

---

### 3️ Transfer Learning Application

* Use of the **VGG16** model pre-trained on **ImageNet**
* Removal of the original classification layer
* Addition of a new `Dense` layer with `Softmax` activation for the selected flower classes
* Freezing of all VGG16 layers, training only the new classification layer

---

### 4️ Model Evaluation

* Monitoring of **validation loss** and **validation accuracy** during training
* Comparison between the model trained from scratch and the Transfer Learning model
* Final evaluation using the test dataset

---

##  Results

* The CNN trained from scratch showed greater variability during validation
* The Transfer Learning model demonstrated:

  * More stable convergence
  * Lower validation loss
  * Higher accuracy, reaching values between **80% and 100%** on the validation set
* The Transfer Learning approach achieved better generalization on unseen data

---

##  Image Prediction

The final model can predict new images by:

* Loading and preprocessing an external image
* Generating class probability scores
* Returning the predicted flower category with the highest probability

---

##  Conclusion

This project demonstrates that **Transfer Learning** is a powerful approach for image classification tasks with limited datasets. By leveraging a pre-trained model such as **VGG16**, it is possible to achieve higher accuracy, faster convergence, and better generalization compared to training a neural network from scratch.

---

##  Future Improvements

* Include more flower categories
* Apply data augmentation techniques
* Perform fine-tuning on deeper VGG16 layers
* Test other pre-trained architectures (ResNet, MobileNet)

---

##  Author

**Wendy Díaz Valdés**
PhD in Pure and Applied Mathematics

