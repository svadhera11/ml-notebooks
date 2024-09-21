# ML Notebooks

This repository contains various implementations of machine learning algorithms, as well as applications of various algorithms on Kaggle Datasets. Below is an overview of the folder structure, including key details of the most important files.

## Table of Contents

- [Implementations](#implementations)
  - [Neural Networks from Scratch](#neural-networks-from-scratch)
  - [GPT Model (Modified from Karpathy)](#gpt-model-modified-from-karpathy)
  - [Logistic Regression](#logistic-regression)
  - [Simple Convolutional Network (JAX)](#simple-convolutional-network-jax)
  - [RNN for Text Generation](#rnn-for-text-generation)
- [Kaggle Projects](#kaggle-projects)
- [Future Work](#future-work)

## Implementations

The `implementations/` folder contains several machine learning models and neural networks built from scratch, following tutorials or developed independently.

### Neural Networks from Scratch

- **`nn-from-scratch.ipynb`** and **`nn-from-scratch-improved.ipynb`**:
  These notebooks implement a basic feedforward neural network from scratch using only Python and NumPy. The improved version includes optimization tweaks such as improvements in weight initialization.

### GPT Model (Modified from Karpathy)

- **`gpt_v3_karpathy.py`**:
  This is a Python implementation of a GPT (Generative Pretrained Transformer) model, based on Andrej Karpathy's code tutorial from [his YouTube video](https://www.youtube.com/watch?v=kCc8FmEb1nY). 

  **Modifications:**
  - The `MultiHeadAttention` has been modified into `CombinedMultiHeadAttention`, which treats the number of heads as a new dimension.
  - Instead of splitting input data with `n_embed` channels into `num_heads` heads, the input is expanded into `num_heads * head_size` channels and then split, providing each head with more information.


### Logistic Regression

- **`LogisticRegression.ipynb`**:
  This notebook implements logistic regression from scratch. 

### Simple Convolutional Network (JAX)

- **`simple-convnet-jax.ipynb`**:
  A simple convolutional neural network (CNN) built using JAX. This notebook demonstrates how to implement convolutional layers and training loops in JAX, a high-performance numerical computing library.

### RNN for Text Generation

- **`simple_rnn_text_generation.py`**:
  This script implements a simple Recurrent Neural Network (RNN) for text generation, showcasing how to train an RNN to generate sequences of text based on a given corpus (Tiny Shakespeare, saved as data/tiny_s.txt).

## Kaggle Projects

The `kaggle/` directory contains a series of notebooks focused on solving problems from various Kaggle competitions or datasets. Please note that many of these are **works-in-progress** and lack final structure or refinement.

Key notebooks include:
- **`titanic-predictions.ipynb`**: Predicting survival on the Titanic using classification models.
- **`mobile_price_classification_logistic.ipynb`**: Logistic regression for predicting mobile price categories.
- **`tumor-size-prediction-tree-regression.ipynb`**: Tree-based regression models for tumor size prediction.
- **`clustering-penguin-species.ipynb`**: Clustering techniques applied to classify different penguin species.

Additional notebooks explore techniques like KMeans clustering, decision trees, Gaussian Mixture Models (GMM), and more.

## Future Work

The files in the `kaggle/` folder are currently under development, with plans for improving their organization (adding comments, markdown notes, and details about algorithms and reasoning why) and performance (hyperparameter choice, regularization, etc.) . Upcoming work will focus on finalizing code structure and enhancing readability.

Stay tuned for further updates!
