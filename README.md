# League of Legends Match Predictor

## Overview

This project implements a logistic regression model using PyTorch to predict the outcomes of League of Legends matches.
It demonstrates comprehensive data loading, preprocessing, model training, hyperparameter tuning, evaluation, and
feature importance analysis.

## Dataset

The dataset used in this project:

- **league_of_legends.csv**

## Project Steps

### Step 1: Data Loading and Preprocessing

- Load dataset using `pandas`
- Split data into training and testing sets (80%-20%)
- Standardize features using `StandardScaler`
- Convert data into PyTorch tensors

### Step 2: Logistic Regression Model

- Define logistic regression model architecture with PyTorch
- Initialize model, loss function (Binary Cross Entropy Loss), and optimizer (Stochastic Gradient Descent)

### Step 3: Training the Model

- Train the logistic regression model for multiple epochs
- Evaluate training performance periodically

### Step 4: Model Optimization

- Apply L2 regularization (weight decay) to prevent overfitting
- Retrain the model and evaluate performance improvements

### Step 5: Model Evaluation and Visualization

- Compute confusion matrix and classification report
- Plot ROC curve and calculate AUC to assess model quality

### Step 6: Save and Load the Model

- Demonstrate techniques to save (`torch.save`) and load (`torch.load`) PyTorch models
- Evaluate the loaded model to confirm consistent performance

### Step 7: Hyperparameter Tuning

- Perform tuning to identify the best learning rate from `[0.01, 0.05, 0.1]`
- Evaluate test accuracy for each learning rate and select the best

### Step 8: Feature Importance

- Extract and visualize model weights to interpret feature significance

## Required Libraries

```bash
pip install pandas scikit-learn torch matplotlib
```

## License

This project is licensed under the MIT License.

