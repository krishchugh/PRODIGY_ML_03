import numpy as np
import pandas as pd
import os
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
import plotly.express as px
import random
from PIL import Image
import seaborn as sns

# Function to unzip dataset files
def extract_files(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

extract_files('Task3/train.zip', 'Task3/working/train')
extract_files('Task3/test1.zip', 'Task3/working/test')

train_dir = 'Task3/working/train/train'
test_dir = 'Task3/working/test/test1'

# Function to read and process images
def fetch_images(directory, target_size=(64, 64), max_images=20, label_filter=None):
    image_data = []
    image_labels = []
    file_list = os.listdir(directory)
    
    if label_filter:
        file_list = [file for file in file_list if label_filter in file]
    
    random.shuffle(file_list)
    for file in file_list[:max_images]:
        try:
            path = os.path.join(directory, file)
            image = Image.open(path).resize(target_size)
            image_array = np.array(image) / 255.0  # Normalize pixel values
            image_data.append(image_array)
            image_labels.append(0 if 'cat' in file else 1)
        except Exception as error:
            print(f"Could not load image {file}: {error}")
    
    return np.array(image_data), np.array(image_labels)

# Function to display images
def display_images(images, labels, num_images=20):
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    axes = axes.flatten()
    
    for i in range(num_images):
        axes[i].imshow(images[i])
        axes[i].set_title('Cat' if labels[i] == 0 else 'Dog')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Load and visualize sample cat images
print("Training Directory:", train_dir)
cats, cat_labels = fetch_images(train_dir, max_images=20, label_filter='cat')
display_images(cats, cat_labels)

# Load and visualize sample dog images
dogs, dog_labels = fetch_images(train_dir, max_images=20, label_filter='dog')
display_images(dogs, dog_labels)

# Load and visualize a mix of cat and dog images
all_images, all_labels = fetch_images(train_dir, max_images=20)
display_images(all_images, all_labels)

# Load a larger dataset for training
images, labels = fetch_images(train_dir, max_images=7500)
print("Images shape:", images.shape)
print("First image shape:", images[0].shape)

# Flatten and scale images
num_samples, img_height, img_width, img_channels = images.shape
images_flat = images.reshape(num_samples, -1)
scaler = StandardScaler()
images_flat_scaled = scaler.fit_transform(images_flat)

# Apply t-SNE for visualization
perplexity_value = min(30, num_samples - 1)
tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)
images_tsne = tsne.fit_transform(images_flat_scaled)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images_flat_scaled, labels, test_size=0.2, random_state=42)
label_counts = dict(zip(*np.unique(labels, return_counts=True)))
print(f"Label distribution: {label_counts}")

# Plot t-SNE results
def plot_tsne(tsne_data, labels, plot_title):
    df_tsne = pd.DataFrame()
    df_tsne['X'] = tsne_data[:, 0]
    df_tsne['Y'] = tsne_data[:, 1]
    df_tsne['Label'] = labels
    df_tsne['Label'] = df_tsne['Label'].map({0: 'Cat', 1: 'Dog'})
    
    fig = px.scatter(df_tsne, x='X', y='Y', color='Label', title=plot_title)
    fig.show()

plot_tsne(images_tsne, labels, 't-SNE of Cat vs Dog Images')

# Train SVM classifier
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Validate the model
y_val_pred = svm_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
val_report = classification_report(y_val, y_val_pred, target_names=['Cat', 'Dog'])
val_conf_matrix = confusion_matrix(y_val, y_val_pred)
print(f'Validation Accuracy: {val_accuracy:.4f}')
print('Classification Report:\n', val_report)

# Display confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(val_conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix - Validation')
plt.show()

# Process and evaluate test data
test_images, test_labels = fetch_images(test_dir)
test_images = test_images / 255.0
num_test_samples = test_images.shape[0]
test_images_flat = test_images.reshape(num_test_samples, -1)
test_images_flat_scaled = scaler.transform(test_images_flat)

# t-SNE for test data
test_perplexity = min(30, num_test_samples - 1)
test_tsne = TSNE(n_components=2, perplexity=test_perplexity, random_state=42)
test_images_tsne = test_tsne.fit_transform(test_images_flat_scaled)

# Predict test data
y_test_pred = svm_model.predict(test_images_flat_scaled)

# Plot t-SNE results for test data
df_test_tsne = pd.DataFrame()
df_test_tsne['X'] = test_images_tsne[:, 0]
df_test_tsne['Y'] = test_images_tsne[:, 1]
df_test_tsne['Predicted Label'] = y_test_pred
df_test_tsne['Predicted Label'] = df_test_tsne['Predicted Label'].map({0: 'Cat', 1: 'Dog'})

fig = px.scatter(df_test_tsne, x='X', y='Y', color='Predicted Label', title='t-SNE of Test Data Predictions')
fig.show()
