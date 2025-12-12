# Deep Learning Projects

This repository documents my journey through deep learning, featuring implementations from fundamental concepts to advanced architectures. The projects are implemented primarily in PyTorch, with some TensorFlow/Keras examples.

## Core Projects

### 1. Fundamentals
- **Lab1.py**: Introduction to PyTorch tensors and basic operations
- **Lab2.py**: Implementing and training a simple neural network
- **Lab3.py**: Working with custom datasets and data loaders
- **Lab4a.py**: Visualizing gradients and backpropagation
- **Lab4b_tensorflowkeras_VisualisingGradients.py**: Gradient visualization using TensorFlow/Keras
- **Lab5.py**: Implementing activation functions and their derivatives

### 2. Advanced Neural Networks
- **Lab6a.py & Lab6b_CustomDataset.py**: Working with custom datasets
- **Lab7.py**: Implementing batch normalization
- **Lab8_***: Series on optimization techniques
  - `Lab8_BatchNormFromScratch.py`: Custom batch normalization
  - `Lab8_dropoutFromScratch.py`: Implementing dropout
  - `Lab8_optimisations.py`: Various optimization algorithms

### 3. Convolutional Neural Networks
- **Lab11_CNN_fromscratch.py**: Building CNNs from scratch
- **Lab12/***: CNN applications
  - `Lab12_MNIST.py`: MNIST classification
  - `Lab12_cifar.py`: CIFAR-10 classification
- **cnn_samplecode.py**: Reference CNN implementation for MNIST

### 4. Recurrent Neural Networks & Transformers
- **Lab14/***: Sequence modeling
  - Character-level language modeling
  - TensorFlow and from-scratch implementations
- **Lab15/***: Advanced RNNs
  - LSTM and vanilla RNN implementations
  - Teacher forcing techniques
- **Lab16/***: Advanced architectures
  - GRU, LSTM, and Transformer implementations
- **Lab17/***: Sequence-to-sequence models
  - LSTM and Transformer-based models

### 5. Generative Models
- **Lab19_GANs.py**: Generative Adversarial Networks implementation
- **Lab19_handwritten_digits_gen.py**: Digit generation using GANs

## Practice Implementations (DL_Practice/)

### Core Concepts
- **FFN_***: Feed-Forward Networks
  - `FFN_fromscratch.py`: Basic implementation
  - `FFN_make_classification.py`: Classification on synthetic data
  - `FFN_MNIST_classifier.py`: MNIST classification

- **CNN_***: Convolutional Neural Networks
  - `CNN_fromscratch.py`: Custom CNN implementation
  - `CNN_cifar10_classifier.py`: CIFAR-10 classification
  - `CNN_Mnist_classifier.py`: MNIST digit classification

### Advanced Topics
- **Seq2Seq_***: Sequence-to-sequence learning
  - `Seq2SeqLearningRNN.py`: RNN-based seq2seq
  - `Seq2Seq_Transformers.py`: Transformer-based seq2seq
- **autoencoder.py**: Implementation of autoencoders
- **vanishing_gradient.py**: Demonstrating and solving vanishing gradients

## Transfer Learning
- **TransferLearning/***: Fine-tuning pre-trained models
  - `Finetuning_linearlayer.py`: Linear layer fine-tuning
  - `FineTuning_FreezingFirstLayers.py`: Layer freezing strategies

## Gene Analysis
Working with gene data including:
- STAT1, MYC, TP53, EGFR, BRCA1, KRAS, ABL1, FOXA2

## Setup & Usage

### Dependencies
- Python 3.8+
- PyTorch & torchvision
- TensorFlow (for specific labs)
- NumPy, Matplotlib, scikit-learn

### Installation
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# Install dependencies
pip install torch torchvision numpy matplotlib scikit-learn
pip install tensorflow  # For TF/Keras labs
```

## Running Examples

### Basic CNN on MNIST
```bash
python cnn_samplecode.py
```

### CIFAR-10 Classification
```bash
python DL_Practice/CNN_cifar10_classifier.py
```

### Transfer Learning
```bash
python TransferLearning/Finetuning_linearlayer.py
```

## Project Structure
```
DeepLearning/
├── DL_Practice/       # Practice implementations
├── Lab*/              # Lab exercises
├── TransferLearning/  # Transfer learning experiments
├── data/              # Dataset storage
└── Theory/            # Course materials
```

## Notes
- Most datasets are downloaded automatically
- Scripts automatically detect and use GPU if available
- Each lab is self-contained with necessary imports and data loading
