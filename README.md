# Cats vs Dogs Image Classifier using CNN

This project is a deep learning-based image classifier that identifies whether an image contains a cat or a dog using a Convolutional Neural Network (CNN). It was developed as part of an internship project at IBM.

## Project Overview

- Objective: Build a binary classifier for cats and dogs using image data.
- Dataset: 25,000 labeled images from the Kaggle Dogs vs Cats dataset.
- Approach:
  - Preprocess images and apply data augmentation.
  - Build and train a CNN model using Keras and TensorFlow.
  - Evaluate the model using accuracy and loss metrics.

## Technologies Used

- Python
- TensorFlow and Keras
- NumPy and Matplotlib
- OpenCV
- Google Colab (for training)

## CNN Architecture

- 4 convolutional layers with ReLU activation
- Max pooling layers
- Dropout for regularization
- Fully connected dense layers
- Sigmoid output layer for binary classification

## Model Performance

- Training Accuracy: XX%
- Validation Accuracy: XX%
- Loss Function: Binary Crossentropy
- Optimizer: Adam
- Epochs: 10
- Batch Size: 32

## Project Structure

cats-vs-dogs-cnn/
├── cats_vs_dogs_cnn.ipynb # Main notebook
├── cats_vs_dogs_cnn.h5 # Trained model file
├── requirements.txt # Dependencies
├── images/ # Sample outputs or graphs
└── README.md # Project documentation


## Sample Prediction

Example:
Input image: dog.1201.jpg  
Predicted: Dog

## How to Run

- **1. Clone the repository:**
   git clone https://github.com/yourusername/cats-vs-dogs-cnn.git
- **2. Install dependencies:**
   pip install -r requirements.txt
- **3. Open the notebook** `cats_vs_dogs_cnn.ipynb` in Jupyter Notebook or Google Colab.
- **4. Run all cells to train and evaluate the model.**

## References

- **Kaggle Dogs vs Cats Dataset:** https://www.kaggle.com/competitions/dogs-vs-cats/data
- **TensorFlow Documentation:** https://www.tensorflow.org/
- **Keras API:** https://keras.io/

## Conclusion

This project demonstrates an end-to-end image classification pipeline using CNNs. It includes data preparation, model design, training, evaluation, and predictions. It was built as a part of a hands-on learning experience during my internship.


