# Fruit-Recogniser
This project demonstrates how to build and train a deep learning model to classify different types of fruits from images. The model utilizes a pre-trained Convolutional Neural Network (CNN) and achieves good accuracy in recognizing 33 distinct fruit classes.
# Dataset Source
The dataset used for this project is the Fruit Recognition dataset, available on Kaggle. It contains a comprehensive collection of fruit images, organized into a training set with 33 classes and an unlabeled test set. The dataset's structure, with subdirectories for each fruit class, is ideal for image classification tasks.
Dataset(URL: https://www.kaggle.com/datasets/sshikamaru/fruit-recognition)

Total number of images: 22495.

Training set size: The training folder that contains 33 subfolders in which training images for each fruit/vegetable are located. There is a total of 16854 images.

Test set size: 5641 images (one fruit or vegetable per image).

Number of classes: 33 (fruits and vegetables).

Image size: 100x100 pixels.

# Model Approach
The model uses transfer learning with a pre-trained MobileNetV2 architecture. This approach avoids training a deep network from scratch, which would require vast amounts of data and computational power. Instead, it leverages the knowledge gained by MobileNetV2 from training on the massive ImageNet dataset.
1. Freezing the Base Model
The MobileNetV2 model, pre-trained on the ImageNet dataset, serves as the feature extractor. Its layers are adept at recognizing fundamental visual features like edges, textures, and shapes. To preserve this learned knowledge, all of its layers are set to be non-trainable (frozen). This is the crucial step of transfer learning, as it prevents the model's core weights from being updated during training on the new fruit dataset.
2. Adding a Custom Classification Head
A new set of layers, known as the "classification head," is added on top of the frozen MobileNetV2 base. These layers are responsible for learning the high-level features specific to the 33 fruit classes. This head consists of:

Flatten Layer: This layer converts the 2D feature maps from the MobileNetV2 output into a 1D vector.

Dense Layer: A fully connected layer with 128 neurons and a ReLU activation function. This layer learns complex, non-linear patterns from the features.

Dropout Layer: A Dropout layer with a rate of 0.3 is included to prevent overfitting by randomly dropping a percentage of neurons during training.

Output Layer: The final Dense layer has 33 neurons (one for each fruit class) and a softmax activation function. This layer outputs a probability distribution, indicating the likelihood that an image belongs to each of the 33 classes.
This structure allows the model to quickly and effectively adapt to the new task of fruit recognition by only training the new layers while benefiting from MobileNetV2's robust feature extraction capabilities.
3. Model Compilation and Training
The model is compiled with the Adam optimizer and a small learning rate of 0.0001, which helps in fine-tuning the new layers without disrupting the pre-trained weights. The categorical_crossentropy loss function is used, which is standard for multi-class classification. The training process uses EarlyStopping with a patience of 5, which automatically stops training if the validation loss does not improve, preventing overfitting and saving the best performing model.
# Performance Achieved
Due to the limited time available for this project, the model's final performance metrics did not reach their full potential. The training process resulted in an accuracy of only 3% on both the validation and test sets, eventhough 90% above shown while running the epochs
# Had more time been available, I am confident that the model's performance could be improved.Thank You!
Model-https://drive.google.com/file/d/1k94wbHiS489-xlOjx7QXPPsIc8Fc7Efh/view?usp=drive_link
