# Optimizing Deep Learning Models for Text-Based Emotion Detection

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red)

A sophisticated deep learning project that combines BERT-based transformers with optimized architectures (including LMU - Legendre Memory Units) for accurate emotion detection from text.

## Project Overview

This project implements an advanced emotion detection system that can classify text into six different emotions:
- Sadness
- Joy
- Love
- Anger
- Fear
- Surprise

The model utilizes state-of-the-art natural language processing techniques and custom neural architectures to achieve high accuracy in emotion classification.

### Key Features

-  Advanced neural architecture combining BERT with Deep LMU
-  Custom mean pooling and attention mechanisms
-  Streamlit-based web interface for easy interaction
-  Support for 6 distinct emotion categories


## Installation

1. Clone the repository:
```bash
git clone https://github.com/Navneeth-Krishna/Optimizing-Deep-Learning-Models-for-Text-Based-Emotion-Detection.git
cd Optimizing-Deep-Learning-Models-for-Text-Based-Emotion-Detection
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Running the Web Interface

To start the Streamlit web interface:

```bash
streamlit run app.py
```

This will launch a web interface where you can input text and get real-time emotion predictions.

## Model Architecture

The project implements a custom architecture that includes:

- BERT-based encoder (boltuix/bert-emotion)
- Deep LMU (Legendre Memory Units) for temporal processing
- Custom mean pooling layer
- Multi-head attention mechanism
- Dropout layers for regularization

```python
class EmotionClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.lmu = DeepLMU(self.bert.config.hidden_size)
        self.pooler = MeanPooling()
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
```

## Project Structure

- `app.py`: Streamlit web application for the emotion detection interface
- `Main.ipynb`: Main training and model development notebook
- `Trail2.ipynb`: Experimental iterations and model improvements
- Various `.pt` files: Saved model checkpoints for different configurations
- `requirements.txt`: Project dependencies

## Dataset

The model is trained on the "dair-ai/emotion" dataset from Hugging Face, which provides a comprehensive collection of text samples labeled with emotions. The dataset is automatically downloaded during the training process.


## Acknowledgments

- dair-ai/emotion dataset for training data
- Hugging Face for transformer models and datasets
- PyTorch team for the deep learning framework
- Streamlit team for the web interface framework

## Author

- [Navneeth Krishna Aravind](https://github.com/Navneeth-Krishna)
