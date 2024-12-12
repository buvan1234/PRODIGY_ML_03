

# Cat and Dog Image Classification using SVM

This project demonstrates how to classify images of cats and dogs from the Kaggle "Cats and Dogs 40" dataset using a Support Vector Machine (SVM). The model uses features extracted from images, including pixel data, to perform classification.

## Overview

This project uses a Support Vector Machine (SVM) to classify images from the "Cats and Dogs 40" Kaggle dataset. The dataset contains images of cats and dogs that are resized, processed, and flattened into feature vectors. SVM is used to build a classification model that can predict whether an image is of a cat or a dog.

## Features

- **Image Classification**: Classify images of cats and dogs.
- **Support Vector Machine (SVM)**: Using SVM with a grid search to fine-tune hyperparameters.
- **Data Preprocessing**: Images are resized and flattened into feature vectors to be used for training.

## Setup Instructions

### Requirements

- Python 3.7+
- TensorFlow
- Keras
- Scikit-learn
- OpenCV
- Matplotlib
- NumPy
- Pandas
- Scikit-image

### Installing Dependencies

1. Clone this repository:
    
    git clone https://github.com/buvan1234/PRODIGY_ML_03.GIT
    

2. Create and activate a virtual environment (optional but recommended):
   
    python3 -m venv env
    source env/bin/activate  # For Mac/Linux
    env\Scripts\activate     # For Windows
  

  

### Dataset

The dataset used in this project is the **Cats and Dogs 40** dataset from Kaggle. You can download the dataset using the Kaggle API.

1. First, ensure that you have the Kaggle API key (`kaggle.json`) saved and available in your working directory. To get the API key, follow the instructions on [Kaggle's official site](https://www.kaggle.com/docs/api).

2. Create a `.kaggle` directory and copy your `kaggle.json` API key into it:
    ```bash
    mkdir -p ~/.kaggle
    cp kaggle.json ~/.kaggle
    ```

3. Download the dataset using the Kaggle API:
    ```bash
    kaggle datasets download -d stefancomanita/cats-and-dogs-40
    ```

4. Extract the downloaded dataset:
    ```bash
    unzip cats-and-dogs-40.zip -d catsAndDogs40
    ```

### Model Overview

1. **Image Preprocessing**: Images are loaded, resized to 40x40 pixels, and flattened to a 1D feature vector.
2. **SVM Classifier**: The Support Vector Machine (SVM) is used for classification. Hyperparameter tuning is done via grid search (`GridSearchCV`).
3. **Model Training**: The SVM model is trained on image features, and performance is evaluated on a test dataset.

### Training and Evaluation

Once the dataset is ready and extracted, you can run the training script. Here's how the SVM model is trained and evaluated:

```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Flatten and prepare the data
flat_data_arr = []
target_arr = []
for category in ['cat', 'dog']:
    path = os.path.join('catsAndDogs40/train', category)
    for img in os.listdir(path):
        img_array = imread(os.path.join(path, img))
        img_resized = resize(img_array, (40, 40, 3))  # Resize images
        flat_data_arr.append(img_resized.flatten())
        target_arr.append(['cat', 'dog'].index(category))

# Convert lists to numpy arrays
flat_data = np.array(flat_data_arr)
target = np.array(target_arr)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(flat_data, target, test_size=0.2, random_state=77, stratify=target)

# Define parameter grid for GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.0001, 0.001, 0.1, 1],
    'kernel': ['rbf', 'poly']
}

# Create SVM classifier
svc = SVC(probability=True)

# Set up GridSearchCV with the specified parameter grid
model = GridSearchCV(svc, param_grid)

# Train the model
model.fit(x_train, y_train)

# Predict using the test set
y_pred = model.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy*100}%")

# Print classification report
print(classification_report(y_test, y_pred, target_names=['cat', 'dog']))
```

### Results

Once the model is trained, the accuracy of the classifier on the test data is printed. The classification report includes precision, recall, and F1-score for both categories (cat and dog).

### Testing Predictions

You can test the model's predictions by passing images through it. For example:

```python
path = 'catsAndDogs40/test/dog/6.jpg'
img = imread(path)
plt.imshow(img)
plt.show()
img_resize = resize(img, (40, 40, 3))
l = [img_resize.flatten()]
print(f"The predicted image is: {['cat', 'dog'][model.predict(l)[0]]}")
```

