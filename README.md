
# CropCare AI – Multimodal Crop Disease Prediction System

## Overview

**CropCare AI** is a **multimodal web-based crop disease prediction system** that leverages **Computer Vision, Natural Language Processing (NLP), and Machine Learning** to detect crop diseases and assess risk levels.

The system predicts crop diseases using:

* 📷 **Leaf images**
* 📝 **Farmer-provided textual descriptions**
* 🌦️ **Weather conditions (temperature, humidity, rainfall)**

## 🚀 Features

* Upload or capture crop leaf images
* Accept farmer symptom descriptions
* Integrates weather data for better predictions
* Predicts crop disease type and risk level
* Provides disease advisory (organic & chemical remedies)
* Prediction history tracking
* Analytics dashboard
* Community discussion forum
* Web-based UI using Flask & HTML/CSS

## 🛠️ Tech Stack

### Frontend
* HTML
* CSS
* JavaScript

### Backend
* Flask (Python)
* Firebase (Community & History Storage)

### Machine Learning & AI
* ResNet50 (Image Feature Extraction)
* BLIP (Image Caption Generation)
* SentenceTransformer (BERT-based Text Embeddings)
* Random Forest Classifier (Final Prediction)

## 🧠 Multimodal Data Preparation

### 1️⃣ Image Captions (Text Modality)

To enrich visual understanding, **automatic captions** were generated for each leaf image using **BLIP (Bootstrapped Language-Image Pretraining)**.

**Example captions:**

* *“Corn leaf showing reddish-brown rust spots indicating early infection.”*
* *“Apple leaf with black circular lesions and yellow margins.”*

These captions help the model associate visual patterns with disease semantics.

### 2️⃣ Weather Data (Environmental Modality)

Since real-time weather data aligned with each image was unavailable, **synthetic but biologically realistic weather data** was generated.

| Parameter   | Unit | Range   |
| ----------- | ---- | ------- |
| Temperature | °C   | 15 – 35 |
| Humidity    | %    | 50 – 90 |
| Rainfall    | mm   | 0 – 100 |

This helps the model learn disease-environment correlations.

## 📈 Model Performance
* **Overall accuracy:** **81.4%**
