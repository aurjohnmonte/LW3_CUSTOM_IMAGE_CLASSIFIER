# LW3_CUSTOM_IMAGE_CLASSIFIER

<h1>THIS IS THE LINK FOR MY GOOGLE COLLAB</h1>
https://colab.research.google.com/drive/1o0AilPAWzaSBxLBi-m0NcEtiJUTZFgtM?usp=sharing

# 🌿 Custom Image Classification using TensorFlow (Google Colab)

## Overview

This project demonstrates how to build an **image classification system using TensorFlow and Convolutional Neural Networks (CNNs)**.

The workflow includes:

* Preparing a custom dataset stored in **Google Drive**
* Loading the dataset in **Google Colab**
* Training a **CNN model**
* Evaluating model performance
* Detecting **overfitting**
* Improving the model using **data augmentation and dropout**
* Predicting new images
* Saving and reusing the trained model

---

# 📂 Project Structure

The dataset must be organized in Google Drive using the following structure:

```
MyDrive/
└── ImageDataset/
    ├── ClassA/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   ├── img3.jpg
    ├── ClassB/
    │   ├── img1.jpg
    │   ├── img2.jpg
    ├── ClassC/
```

📌 **Important Notes**

* Each folder represents a **class label**
* Minimum **20 plant species categories**
* Each category should have **at least 250 images**

TensorFlow automatically uses **folder names as labels**.

---

# 🚀 Part 1: Preparing and Loading Custom Images

## Step 1: Prepare Image Dataset

1. Collect images of **20 plant species**
2. Ensure each category has **≥250 images**
3. Organize images using the folder structure above

---

## Step 2: Upload Dataset to Google Drive

1. Open **Google Drive**
2. Upload the `ImageDataset` folder
3. Copy the folder link

Example:

```
https://drive.google.com/drive/folders/your-folder-id
```

---

## Step 3: Open Google Colab

1. Go to **Google Colab**
2. Create a **New Notebook**

---

## Step 4: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

Authorize access when prompted.

---

## Step 5: Define Dataset Path

```python
dataset_path = "/content/drive/MyDrive/ImageDataset"
```

---

## Step 6: Load Images Using TensorFlow

```python
import tensorflow as tf

img_height = 180
img_width = 180
batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
```

---

## Step 7: View Class Names

```python
class_names = train_ds.class_names
print(class_names)
```

---

# 🧠 Part 2: Training the CNN Model

## Step 1: Optimize Dataset Performance

```python
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
```

---

## Step 2: Build CNN Model

```python
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),

    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names))
])
```

---

## Step 3: Compile Model

```python
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
```

---

## Step 4: Train the Model

```python
epochs = 10

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)
```

---

## Step 5: Evaluate Model

```python
loss, accuracy = model.evaluate(val_ds)

print("Validation Accuracy:", accuracy)
```

---

## Step 6: Test With New Image

Upload a test image to Google Drive.

```python
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

img_path = "/content/drive/MyDrive/test.jpg"

img = load_img(img_path, target_size=(img_height, img_width))
img_array = img_to_array(img)

img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print("Predicted Class:", class_names[np.argmax(score)])
```

---

# 📊 Part 3: Visualizing Training Results

```python
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training vs Validation Accuracy')

plt.subplot(1,2,2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training vs Validation Loss')

plt.show()
```

---

# ⚠️ Detecting Overfitting

| Sign                                      | Meaning      |
| ----------------------------------------- | ------------ |
| Training accuracy ↑ Validation accuracy ↓ | Overfitting  |
| Both accuracies low                       | Underfitting |
| Both accuracies high and close            | Good fit     |

---

# 🔄 Part 4: Data Augmentation

## Step 1: Create Augmentation Layer

```python
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])
```

---

## Step 2: Visualize Augmented Images

```python
import matplotlib.pyplot as plt

for images, _ in train_ds.take(1):
    plt.figure(figsize=(8,8))

    for i in range(9):
        augmented_images = data_augmentation(images)

        ax = plt.subplot(3,3,i+1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")
```

---

# 🛡 Part 5: Reducing Overfitting Using Dropout

```python
model = models.Sequential([
    data_augmentation,

    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),

    layers.Conv2D(16,3,padding='same',activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(32,3,padding='same',activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64,3,padding='same',activation='relu'),
    layers.MaxPooling2D(),

    layers.Dropout(0.3),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),

    layers.Dropout(0.3),

    layers.Dense(len(class_names))
])
```

---

# 🔁 Part 6: Retrain Improved Model

```python
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

epochs = 15

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)
```

---

# 🔎 Part 7: Predict New Images

```python
img_path = "/content/drive/MyDrive/test_image.jpg"

img = load_img(img_path, target_size=(img_height, img_width))
img_array = img_to_array(img)

img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print("Predicted Class:", class_names[np.argmax(score)])
print("Confidence:", round(100*np.max(score),2),"%")
```

---

# 💾 Part 8: Save and Reuse Model

## Save Model

```python
model.save("/content/drive/MyDrive/my_image_classifier")
```

---

## Load Model

```python
from tensorflow.keras.models import load_model

loaded_model = load_model("/content/drive/MyDrive/my_image_classifier")
```

---

# 📌 Key Concepts Learned

* Image dataset preparation
* CNN architecture for classification
* Model training and evaluation
* Detecting **overfitting**
* **Data augmentation**
* **Dropout regularization**
* Model deployment preparation

---

# 🌍 Possible Applications

This image classification system can be applied to:

* 🌿 **Plant species identification**
* 🌾 **Smart agriculture monitoring**
* 📱 **Mobile plant recognition apps**
* 🌳 **Environmental biodiversity research**

---

# 🛠 Technologies Used

* Python
* TensorFlow / Keras
* Google Colab
* Google Drive
* Matplotlib

---

# 👨‍💻 Author

Student Laboratory Exercise
Custom Image Classification using TensorFlow

---
