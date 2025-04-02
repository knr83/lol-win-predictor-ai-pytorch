# ==========================================================
# League of Legends Match Outcome Predictor using PyTorch
# ==========================================================

import itertools

import matplotlib.pyplot as plt
# SECTION 1: Import Libraries
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# SECTION 2: Data Loading and Preprocessing
# Load dataset from CSV file
data = pd.read_csv('lol_data.csv')

# Split data into features (X) and target (y)
X = data.drop('win', axis=1)
y = data['win']

# Split dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features to zero mean and unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)


# SECTION 3: Define Logistic Regression Model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


# Initialize model, loss function, and optimizer
input_dim = X_train.shape[1]
model = LogisticRegressionModel(input_dim)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)

# SECTION 4: Train Model with L2 Regularization
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    # Print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# SECTION 5: Evaluate Model
model.eval()
with torch.no_grad():
    # Predict on test set
    y_pred_test = model(X_test_tensor).numpy()
    y_pred_test_labels = (y_pred_test > 0.5).astype(int)
    y_test_np = y_test_tensor.numpy()

    # Compute accuracy
    accuracy = (y_pred_test_labels == y_test_np).mean()
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

# SECTION 6: Visualization (Confusion Matrix, ROC Curve)
# Confusion Matrix visualization
cm = confusion_matrix(y_test_np, y_pred_test_labels)
plt.figure(figsize=(6, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = range(2)
plt.xticks(tick_marks, ['Loss', 'Win'], rotation=45)
plt.yticks(tick_marks, ['Loss', 'Win'])

# Annotate confusion matrix
thresh = cm.max() / 2
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Classification report
print("Classification Report:\n", classification_report(y_test_np, y_pred_test_labels, target_names=['Loss', 'Win']))

# ROC Curve visualization
fpr, tpr, _ = roc_curve(y_test_np, y_pred_test)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# SECTION 7: Save and Load the Trained Model
# Save trained model parameters
torch.save(model.state_dict(), 'logistic_model.pth')

# Load model parameters into a new model instance
loaded_model = LogisticRegressionModel(input_dim)
loaded_model.load_state_dict(torch.load('logistic_model.pth'))
loaded_model.eval()

# Verify loaded model's accuracy
with torch.no_grad():
    y_loaded = loaded_model(X_test_tensor).numpy()
    y_loaded_labels = (y_loaded > 0.5).astype(int)
    loaded_accuracy = (y_loaded_labels == y_test_np).mean()
    print(f'Loaded Model Test Accuracy: {loaded_accuracy * 100:.2f}%')

# SECTION 8: Hyperparameter Tuning for Learning Rate
learning_rates = [0.01, 0.05, 0.1]
best_accuracy = 0.0
best_lr = None

for lr in learning_rates:
    # Reinitialize model and optimizer
    model = LogisticRegressionModel(input_dim)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.01)

    # Train the model
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    # Evaluate model performance
    model.eval()
    with torch.no_grad():
        preds: object = model(X_test_tensor).numpy()
        preds_labels = (preds > 0.5).astype(int)
        acc = (preds_labels == y_test_np).mean()

    print(f'LR: {lr}, Test Accuracy: {acc * 100:.2f}%')

    # Track best learning rate
    if acc > best_accuracy:
        best_accuracy = acc
        best_lr = lr

print(f'Best Learning Rate: {best_lr}, Accuracy: {best_accuracy * 100:.2f}%')

# SECTION 9: Feature Importance Analysis
# Extract model weights (feature importances)
weights = model.linear.weight.data.numpy().flatten()
features = X.columns

# Create and sort DataFrame of features by importance
feature_importance = pd.DataFrame({'Feature': features, 'Importance': weights})
feature_importance = feature_importance.reindex(
    feature_importance['Importance'].abs().sort_values(ascending=False).index)

# Display feature importances
print(feature_importance)

# Visualize feature importances
plt.figure(figsize=(10, 6))
plt.bar(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
