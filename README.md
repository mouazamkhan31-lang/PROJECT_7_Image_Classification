# 🖼️ Advanced Image Classification and Object Recognition System

## 📌 Project Overview

This project implements an advanced image classification and object recognition system using Deep Learning and Convolutional Neural Networks (CNNs).

The project includes a CNN model built from scratch, EfficientNetB0 transfer learning, fine-tuning, data augmentation, comprehensive model evaluation, model interpretation techniques, a Flask API, TensorFlow Lite deployment, and a Streamlit web application.

The system is trained and evaluated on the CIFAR-10 dataset containing 10 different image classes.

---

## 🎯 Objectives

- Build a CNN model from scratch
- Implement transfer learning using EfficientNetB0
- Fine-tune a pre-trained deep learning model
- Apply data augmentation techniques
- Evaluate model performance using multiple metrics
- Generate confusion matrices and ROC curves
- Implement Grad-CAM visualization
- Generate feature maps and saliency maps
- Build a web-based image classification application
- Create a Flask REST API
- Implement batch image processing
- Convert the model to TensorFlow Lite
- Benchmark inference speed

---

## 🧠 Dataset

The project uses the **CIFAR-10 dataset**.

CIFAR-10 contains 60,000 color images divided into 10 classes:

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

---

## 🏗️ Project Architecture

```text
CIFAR-10 Dataset
        │
        ▼
Data Preprocessing
        │
        ▼
Data Augmentation
        │
        ├───────────────┐
        ▼               ▼
CNN From Scratch    EfficientNetB0
        │               │
        │               ▼
        │          Fine-Tuning
        │               │
        └───────┬───────┘
                ▼
        Model Evaluation
                │
        ┌───────┼────────┐
        ▼       ▼        ▼
   Confusion   ROC    Advanced
    Matrix    Curves   Metrics
                │
                ▼
        Model Interpretation
                │
        ┌───────┼──────────┐
        ▼       ▼          ▼
    Grad-CAM Feature Maps Saliency
                │
                ▼
          Model Deployment
                │
        ┌───────┼────────┐
        ▼       ▼        ▼
    Streamlit Flask   TensorFlow
       App     API       Lite