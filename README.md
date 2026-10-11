# Galaxy Zoo: Hierarchical vs End-to-End CNN

## Project Overview

This AIML339 capstone project investigates whether a simplified hierarchical convolutional neural network (CNN), inspired by the Galaxy Zoo 2 decision tree, provides advantages over an equivalent end-to-end CNN for galaxy morphology classification.

The project compares predictive performance and investigates how classification errors propagate through hierarchical decisions.

## Dataset

This project uses the Galaxy Zoo dataset from the [Galaxy Zoo Challenge on Kaggle](https://www.kaggle.com/competitions/galaxy-zoo-the-galaxy-challenge/data).

The dataset is not included in this repository. Download the training image archive and training solutions separately.

The current notebook expects the dataset paths to be configured before execution.

## Dependencies

- Python 3.11
- PyTorch
- torchvision
- NumPy
- pandas
- scikit-learn
- matplotlib
- Pillow
- Jupyter

A CUDA-compatible GPU is recommended but not required.

## Running the Project

Open `AIML339 - Project Notebook.ipynb` in Jupyter or Positron.

1. Activate the `galaxy-zoo` Conda environment or install the required dependencies.
2. Download the Galaxy Zoo training images and labels.
3. Update the dataset paths in the notebook.
4. Run the notebook cells in order to load the data, train the CNN and evaluate its predictions.

## Current Implementation

The initial implementation includes:

- Loading Galaxy Zoo images directly from a ZIP archive
- Selecting a 5,000-image subset
- Splitting the data into training, validation and test sets
- Resizing images to 64 × 64 pixels
- Training a prototype CNN to predict 37 Galaxy Zoo response values
- Calculating validation and test mean squared error

## Planned Experiments

- Define a simplified Galaxy Zoo classification hierarchy
- Implement an end-to-end CNN baseline
- Implement a hierarchical CNN
- Compare accuracy, precision, recall and F1-score
- Investigate hierarchical error propagation

## Reproducibility

The initial experiments use a random seed of 42 for dataset splitting and a batch size of 32.

Final experiment configurations and results will be documented as development progresses.

## Data and Attribution

Galaxy Zoo data should be obtained from the original distribution source. Third-party reference implementations and downloaded material are not redistributed in this repository.