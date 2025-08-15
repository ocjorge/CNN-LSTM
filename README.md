# EMG Gesture Recognition using CNN-LSTM Hybrid Model

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

A hybrid CNN-LSTM model for sEMG-based gesture recognition using the NinaPro DB1 dataset.

## 📌 Overview

This repository contains a deep learning pipeline for surface electromyography (sEMG) signal classification using a hybrid Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) architecture. The model is trained on the [NinaPro DB1 dataset](http://ninapro.hevs.ch/) to recognize hand gestures from EMG signals.

![Model Architecture](https://via.placeholder.com/800x400.png?text=CNN-LSTM+Architecture+Diagram)

## ✨ Features

- **Hybrid Architecture**: Combines CNN's feature extraction with LSTM's temporal modeling
- **Data Preprocessing**: Sliding window approach with Z-score normalization
- **Modular Design**: Clean separation of data loading, preprocessing, and model building
- **Visualization**: Training history and performance metrics plotting

## 📊 Dataset

The model uses the [NinaPro DB1 dataset](http://ninapro.hevs.ch/) which contains:
- 27 healthy subjects
- 52 hand gestures + rest position
- 10 EMG electrodes at 100Hz sampling rate

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/emg-gesture-recognition.git
cd emg-gesture-recognition
