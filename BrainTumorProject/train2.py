# import numpy as np
# import os
# import cv2
# import tensorflow as tf
# from sklearn.utils import shuffle
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout
# import time


# # -------------------------------
# # PARAMETERS
# # -------------------------------
# image_size = 128
# labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

# train_path = "dataset/Training"
# test_path = "dataset/Testing"

# # -------------------------------
# # PREPROCESS FUNCTION
# # -------------------------------
# def preprocess(img):
#     img = cv2.resize(img, (image_size, image_size))
#     img = cv2.GaussianBlur(img, (3,3), 0)
#     img = img.astype('float32') / 255.0
#     return img

# # -------------------------------
# # LOAD DATA FUNCTION
# # -------------------------------
# def load_data(path):
#     X = []
#     Y = []

#     for label in labels:
#         folderPath = os.path.join(path, label)

#         if not os.path.exists(folderPath):
#             continue

#         for file in os.listdir(folderPath):

#             if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
#                 continue

#             img_path = os.path.join(folderPath, file)
#             img = cv2.imread(img_path)

#             if img is None:
#                 continue

#             img = preprocess(img)

#             X.append(img)
#             Y.append(label)

#     return np.array(X, dtype='float32'), np.array(Y)

# # -------------------------------
# # LOAD DATA
# # -------------------------------
# print("Loading data...")
# X_train, Y_train = load_data(train_path)
# X_test, Y_test = load_data(test_path)

# print("X_train shape:", X_train.shape)
# print("X_test shape:", X_test.shape)
# print("Pixel range:", X_train.min(), X_train.max())

# # -------------------------------
# # SHUFFLE
# # -------------------------------
# X_train, Y_train = shuffle(X_train, Y_train, random_state=42)

# # -------------------------------
# # LABEL ENCODING
# # -------------------------------
# label_to_index = {label: i for i, label in enumerate(labels)}

# Y_train = tf.keras.utils.to_categorical([label_to_index[i] for i in Y_train])
# Y_test = tf.keras.utils.to_categorical([label_to_index[i] for i in Y_test])

# # -------------------------------
# # MODEL
# # -------------------------------
# model = Sequential()

# model.add(Conv2D(32,(3,3),activation='relu',input_shape=(128,128,3)))
# model.add(Conv2D(64,(3,3),activation='relu'))
# model.add(MaxPooling2D(2,2))

# model.add(Conv2D(64,(3,3),activation='relu'))
# model.add(Conv2D(64,(3,3),activation='relu'))
# model.add(MaxPooling2D(2,2))

# model.add(Conv2D(128,(3,3),activation='relu'))
# model.add(Conv2D(128,(3,3),activation='relu'))
# model.add(Conv2D(128,(3,3),activation='relu'))
# model.add(MaxPooling2D(2,2))

# model.add(Conv2D(128,(3,3),activation='relu'))
# model.add(Conv2D(256,(3,3),activation='relu'))
# model.add(MaxPooling2D(2,2))
# model.add(Dropout(0.3))

# model.add(Flatten())

# model.add(Dense(512,activation='relu'))
# model.add(Dense(128,activation='relu'))
# model.add(Dropout(0.4))
# model.add(Dense(4,activation='softmax'))


# model.summary()


# # -------------------------------
# # COMPILE
# # -------------------------------
# model.compile(
#     loss='categorical_crossentropy',
#     optimizer='adam',
#     metrics=['accuracy']
# )

# # -------------------------------
# # TRAIN
# # -------------------------------
# start_time = time.time()

# history = model.fit(
#     X_train, Y_train,
#     epochs=25,
#     batch_size=16,
#     validation_split=0.2,   # correct usage
#     verbose=1
# )

# end_time = time.time()
# training_time = end_time - start_time

# print(f"\nTraining time: {training_time:.2f} sec ({training_time/60:.2f} min)")

# # -------------------------------
# # TEST
# # -------------------------------
# test_loss, test_acc = model.evaluate(X_test, Y_test)
# print(f"\nTest Accuracy: {test_acc:.4f}")

# # -------------------------------
# # SAVE MODEL
# # -------------------------------
# model.save("final_model.h5")

# # -------------------------------
# # FINAL RESULTS
# # -------------------------------
# print("\nFinal Training Accuracy:", history.history['accuracy'][-1])
# print("Final Validation Accuracy:", history.history['val_accuracy'][-1])



# import numpy as np
# import os
# import cv2
# import tensorflow as tf
# from sklearn.utils import shuffle
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import time

# # -------------------------------
# # PARAMETERS
# # -------------------------------
# image_size = 128
# labels = ['glioma', 'meningioma','notumor','pituitary']

# train_path = "dataset/Training"
# test_path = "dataset/Testing"

# # -------------------------------
# # PREPROCESS FUNCTION (FINAL)
# # -------------------------------
# def preprocess(img):
#     img = cv2.resize(img, (image_size, image_size))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # 🔥 important
#     img = img.astype('float32') / 255.0
#     return img

# # -------------------------------
# # LOAD DATA FUNCTION
# # -------------------------------
# def load_data(path):
#     X = []
#     Y = []

#     for label in labels:
#         folderPath = os.path.join(path, label)

#         for file in os.listdir(folderPath):
#             img_path = os.path.join(folderPath, file)
#             img = cv2.imread(img_path)

#             if img is None:
#                 continue

#             img = preprocess(img)

#             X.append(img)
#             Y.append(label)

#     return np.array(X, dtype='float32'), np.array(Y)

# # -------------------------------
# # LOAD DATA
# # -------------------------------
# print("Loading dataset...")
# X_train, Y_train = load_data(train_path)
# X_test, Y_test = load_data(test_path)

# print("Training samples:", len(X_train))
# print("Testing samples:", len(X_test))

# from collections import Counter

# print("Train label distribution:", Counter(Y_train))
# print("Test label distribution:", Counter(Y_test))

# # -------------------------------
# # SHUFFLE
# # -------------------------------
# X_train, Y_train = shuffle(X_train, Y_train, random_state=42)

# # -------------------------------
# # LABEL ENCODING
# # -------------------------------
# Y_train = tf.keras.utils.to_categorical([labels.index(i) for i in Y_train])
# Y_test = tf.keras.utils.to_categorical([labels.index(i) for i in Y_test])

# # -------------------------------
# # DATA AUGMENTATION (LIGHT)
# # -------------------------------
# datagen = ImageDataGenerator(
#     rotation_range=10,
#     zoom_range=0.1,
#     width_shift_range=0.05,
#     height_shift_range=0.05,
#     horizontal_flip=False
# )

# # -------------------------------
# # MODEL (CNN)
# # -------------------------------
# model = Sequential()

# model.add(Conv2D(32,(3,3),activation='relu',input_shape=(128,128,3)))
# model.add(Conv2D(64,(3,3),activation='relu'))
# model.add(MaxPooling2D(2,2))
# model.add(Dropout(0.3))

# model.add(Conv2D(64,(3,3),activation='relu'))
# model.add(Conv2D(64,(3,3),activation='relu'))
# model.add(Dropout(0.3))
# model.add(MaxPooling2D(2,2))
# model.add(Dropout(0.3))

# model.add(Conv2D(128,(3,3),activation='relu'))
# model.add(Conv2D(128,(3,3),activation='relu'))
# model.add(Conv2D(128,(3,3),activation='relu'))
# model.add(MaxPooling2D(2,2))
# model.add(Dropout(0.3))

# model.add(Conv2D(128,(3,3),activation='relu'))
# model.add(Conv2D(256,(3,3),activation='relu'))
# model.add(MaxPooling2D(2,2))
# model.add(Dropout(0.3))
# model.add(Dropout(0.3))

# model.add(Flatten())

# model.add(Dense(512,activation='relu'))
# model.add(Dense(512,activation='relu'))
# model.add(Dropout(0.3))

# model.add(Dense(4,activation='softmax'))

# model.summary()

# # -------------------------------
# # COMPILE
# # -------------------------------
# model.compile(
#     loss='categorical_crossentropy',
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
#     metrics=['accuracy']
# )

# # -------------------------------
# # TRAIN (FINAL)
# # -------------------------------
# print("\nTraining started...")
# start_time = time.time()

# batch_size = 16
# import math
# steps_per_epoch = math.ceil(len(X_train) / batch_size)
# validation_steps = math.ceil(len(X_test) / batch_size)

# datagen.fit(X_train)

# history = model.fit(
#     datagen.flow(X_train, Y_train, batch_size=batch_size),
#     steps_per_epoch=steps_per_epoch, 
#     epochs=25,
#     validation_data=(X_test, Y_test)
# )

# end_time = time.time()

# training_time = end_time - start_time

# print("\nTotal training time:", training_time, "seconds")
# print("Training time (minutes):", training_time/60)
# print(f"\nTraining time: {training_time:.2f} sec ({training_time/60:.2f} min)")

# # -------------------------------
# # FINAL ACCURACY
# # -------------------------------
# print("\nFinal Training Accuracy:", history.history['accuracy'][-1])
# print("Final Validation Accuracy:", history.history['val_accuracy'][-1])

# # -------------------------------
# # TEST ACCURACY
# # -------------------------------
# test_loss, test_acc = model.evaluate(X_test, Y_test)
# print("\nTest Accuracy:", test_acc)

# # -------------------------------
# # SAVE MODEL
# # -------------------------------
# model.save("model.h5")
# print("\nModel saved as model.h5")


import numpy as np
import os
import cv2
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
import math

# -------------------------------
# PARAMETERS
# -------------------------------
image_size = 128
batch_size = 16
labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

train_path = "dataset/Training"
test_path = "dataset/Testing"

# -------------------------------
# PREPROCESS FUNCTION
# -------------------------------
def preprocess(img):
    img = cv2.resize(img, (image_size, image_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # ✅ important
    img = img.astype('float32') / 255.0
    return img

# -------------------------------
# LOAD DATA FUNCTION
# -------------------------------
def load_data(path):
    X = []
    Y = []

    for label in labels:
        folderPath = os.path.join(path, label)

        if not os.path.exists(folderPath):
            continue

        for file in os.listdir(folderPath):

            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            img_path = os.path.join(folderPath, file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            img = preprocess(img)

            X.append(img)
            Y.append(label)

    return np.array(X, dtype='float32'), np.array(Y)

# -------------------------------
# LOAD DATA
# -------------------------------
print("Loading data...")
X_train, Y_train = load_data(train_path)
X_test, Y_test = load_data(test_path)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

from collections import Counter

print("Train distribution:", Counter(Y_train))

# -------------------------------
# SHUFFLE
# -------------------------------
X_train, Y_train = shuffle(X_train, Y_train, random_state=42)

# -------------------------------
# LABEL ENCODING
# -------------------------------
label_to_index = {label: i for i, label in enumerate(labels)}

Y_train = tf.keras.utils.to_categorical([label_to_index[i] for i in Y_train])
Y_test = tf.keras.utils.to_categorical([label_to_index[i] for i in Y_test])

# # -------------------------------
# # DATA AUGMENTATION
# # -------------------------------
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=False   # ❗ good for MRI
)
# datagen = ImageDataGenerator(
#     rotation_range=5,
#     brightness_range=[0.98, 1.05]
# )
datagen.fit(X_train)

# -------------------------------
# MODEL
# -------------------------------
model = Sequential()

model.add(Conv2D(32,(3,3),activation='relu',input_shape=(128,128,3)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(Conv2D(256,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(512,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(4,activation='softmax'))

model.summary()

# -------------------------------
# COMPILE
# -------------------------------
model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    metrics=['accuracy']
)

# -------------------------------
# TRAIN
# -------------------------------
print("\nTraining started...")
start_time = time.time()

steps_per_epoch = math.ceil(len(X_train) / batch_size)

history = model.fit(
    datagen.flow(X_train, Y_train, batch_size=batch_size),
    steps_per_epoch=steps_per_epoch,
    epochs=25,
    validation_data=(X_test, Y_test)
)
# history = model.fit(
#     X_train, Y_train,
#     batch_size=batch_size,
#     epochs=25,
#     validation_data=(X_test, Y_test)
# )

end_time = time.time()
training_time = end_time - start_time

print(f"\nTraining time: {training_time:.2f} sec ({training_time/60:.2f} min)")

# -------------------------------
# TEST
# -------------------------------
test_loss, test_acc = model.evaluate(X_test, Y_test)
print(f"\nTest Accuracy: {test_acc:.4f}")

# -------------------------------
# SAVE MODEL
# -------------------------------
model.save("final_model.h5")

# -------------------------------
# FINAL RESULTS
# -------------------------------
print("\nFinal Training Accuracy:", history.history['accuracy'][-1])
print("Final Validation Accuracy:", history.history['val_accuracy'][-1])