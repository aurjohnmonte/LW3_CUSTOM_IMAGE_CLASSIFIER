# LW3_CUSTOM_IMAGE_CLASSIFIER

<h1>THIS IS THE LINK FOR MY GOOGLE COLLAB</h1>
https://colab.research.google.com/drive/1o0AilPAWzaSBxLBi-m0NcEtiJUTZFgtM?usp=sharing


# 📘 Guide Questions – Answers

## Custom Image Classification using TensorFlow

---

# 🌿 Part 1: Dataset Preparation

## 1. How did you organize your dataset in Google Drive?

I organized the dataset in Google Drive by creating a main folder called **ImageDataset** inside **MyDrive**. Inside this folder, I created separate subfolders for each plant species category. Each folder contains images that belong only to that specific plant species.

### Example Structure

```text
MyDrive/
 └── ImageDataset/
     ├── Rose/
     │   ├── img1.jpg
     │   ├── img2.jpg
     │   ├── img3.jpg
     ├── Sunflower/
     │   ├── img1.jpg
     │   ├── img2.jpg
     ├── Tulip/
     │   ├── img1.jpg
     │   ├── img2.jpg
```

Each folder represents a **class label**, and the images inside it serve as the **training examples** for that class.

---

## 2. Why is folder structure important for TensorFlow image loading?

The folder structure is important because TensorFlow automatically **uses folder names as labels** when loading images using:

```python
image_dataset_from_directory()
```

### Example

```
Rose/ → label: Rose  
Sunflower/ → label: Sunflower  
Tulip/ → label: Tulip
```

This allows TensorFlow to **automatically assign labels to images without manual labeling**, making dataset loading easier and more efficient.

---

# 🧠 Part 2: Model Training

## 3. What is the role of convolutional layers in image classification?

Convolutional layers are responsible for **extracting important visual features from images**. These layers scan the image using filters to detect patterns such as:

* edges
* textures
* shapes
* object parts

In the early layers, the model detects simple features like **edges and lines**, while deeper layers detect **more complex patterns** such as leaves, petals, or plant structures. These extracted features help the model correctly identify the image category.

---

## 4. Why do we split data into training and validation sets?

The dataset is split into **training and validation sets** to properly evaluate the model.

| Dataset        | Purpose                                        |
| -------------- | ---------------------------------------------- |
| Training Set   | Used to train the model and update its weights |
| Validation Set | Used to test model performance on unseen data  |

This approach helps detect problems such as **overfitting**, where the model performs well on training data but poorly on new data.

---

# 📊 Part 3: Performance Analysis

## 5. What accuracy did your model achieve?

After training the model, the validation accuracy achieved was approximately:

```
Validation Accuracy: 0.82
```

This indicates that the model was able to correctly classify most of the plant images in the validation dataset.

*(Note: Replace this value with your actual model result.)*

---

## 6. How did the number of images affect the model’s performance?

The number of images significantly affects the model’s performance.

* **More images → better training → higher accuracy**
* **Fewer images → limited learning → lower accuracy**

A larger dataset allows the model to learn **more variations of plant images**, including different lighting conditions, backgrounds, and angles. This improves the model’s ability to generalize to new images.

---

# 💡 Part 4: Critical Thinking

## 7. What challenges did you encounter while using your own dataset?

Some challenges encountered include:

* Collecting a large number of images for each plant category
* Ensuring images are clear and correctly labeled
* Managing large file sizes when uploading to Google Drive
* Images having different resolutions or backgrounds

Another challenge is ensuring that **each class has a balanced number of images** so the model does not become biased toward one category.

---

## 8. How can data augmentation improve your model?

Data augmentation improves the model by **artificially increasing the variety of training images**.

Common techniques include:

* flipping images
* rotating images
* zooming
* shifting

These transformations create new variations of existing images, helping the model learn **different perspectives of the same object**. This improves generalization and reduces overfitting.

---

# 📉 Part 5: Visualization & Overfitting

## 9. What signs indicated overfitting in your first model?

Overfitting was indicated when:

* Training accuracy continued to **increase**
* Validation accuracy **stopped improving or decreased**
* Validation loss **increased while training loss decreased**

This means the model was **memorizing the training data instead of learning general patterns**.

---

# ⚙️ Part 6: Model Improvement

## 10. What is the purpose of dropout layers?

Dropout layers help **reduce overfitting** by randomly disabling some neurons during training.

This forces the model to:

* avoid relying on specific neurons
* learn more generalized patterns

As a result, the model becomes **more robust and performs better on unseen data**.

---

## 11. Why does data augmentation improve generalization?

Data augmentation improves generalization because it exposes the model to **different variations of the same images**.

Examples include:

* rotated plants
* flipped leaves
* zoomed flowers

This helps the model learn the **essential features of the object rather than memorizing specific images**, allowing it to perform better on new data.

---

# 📈 Part 7: Performance Comparison

## 12. Compare accuracy before and after improvements.

Before applying improvements, the model achieved approximately:

```
Validation Accuracy: 0.78
```

After applying:

* data augmentation
* dropout layers
* additional training epochs

the validation accuracy improved to approximately:

```
Validation Accuracy: 0.86
```

This indicates **better generalization and reduced overfitting**.

*(Replace these values with your actual results.)*

---

## 13. Which technique contributed most to improvement?

Data augmentation contributed the most improvement because it **increased dataset variability** and helped the model learn from more diverse image patterns.

Dropout also helped by **reducing overfitting and improving model stability**.

---

# 🚀 Part 8: Deployment & Application

## 14. Why is saving the model important?

Saving the model is important because it allows the trained model to **be reused without retraining**.

Benefits include:

* deploying the model in applications
* testing new images later
* sharing the model with other systems

The saved model contains both the **architecture and trained weights**.

---

## 15. How can this model be deployed in a real-world system?

The trained model can be deployed in several ways.

### 📱 Mobile Application

Farmers or students can take pictures of plants using their phones, and the application predicts the plant species.

### 🌐 Web Application

Users upload plant images to a website, and the server processes the image and returns the predicted species.

### 🌾 Agriculture Monitoring System

The model can be used in **smart farming systems** to automatically identify plant species or detect crops.

The saved TensorFlow model can be integrated into:

* mobile apps
* web APIs
* cloud-based systems

---

# 📌 Summary

Through this activity, we learned how to:

* Prepare and organize image datasets
* Train a CNN model for image classification
* Evaluate model performance
* Detect and reduce overfitting
* Improve model accuracy using data augmentation and dropout
* Deploy trained models for real-world applications

This project demonstrates the **practical workflow of building an AI-powered image classification system** using TensorFlow.
