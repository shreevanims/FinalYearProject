# import numpy as np
# import os
# import cv2
# import tensorflow as tf
# from sklearn.utils import shuffle
# from keras.models import Sequential
# from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout
# import time
# import matplotlib.pyplot as plt

# # -------------------------------
# # PARAMETERS
# # -------------------------------
# image_size = 128
# labels = ['glioma', 'meningioma','notumor','pituitary']

# train_path = "dataset/Training"
# test_path = "dataset/Testing"

# # -------------------------------
# # PREPROCESS FUNCTION (IMPORTANT)
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

#         for file in os.listdir(folderPath):
#             img = cv2.imread(os.path.join(folderPath, file))

#             if img is None:
#                 continue

#             img = preprocess(img)

#             X.append(img)
#             Y.append(label)

#     return np.array(X, dtype='float32'), np.array(Y)

# # -------------------------------
# # LOAD DATA
# # -------------------------------
# X_train, Y_train = load_data(train_path)
# X_test, Y_test = load_data(test_path)

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
# # MODEL (FULL)
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
#     optimizer='Adam',
#     metrics=['accuracy']
# )

# # -------------------------------
# # TRAIN
# # -------------------------------
# start_time = time.time()

# history = model.fit(
#     X_train, Y_train,
#     epochs=2,
#     batch_size=16,
#     validation_data=(X_test, Y_test)
# )

# end_time = time.time()

# training_time = end_time - start_time

# print("\nTotal training time:", training_time, "seconds")
# print("Training time (minutes):", training_time/60)


# # -------------------------------
# # SAVE MODEL (ADD THIS)
# # -------------------------------
# model.save("model.h5")
# print("Model saved successfully!")

# # -------------------------------
# # PRINT ACCURACY
# # -------------------------------
# print("\nFinal Training Accuracy:", history.history['accuracy'][-1])
# print("Final Validation Accuracy:", history.history['val_accuracy'][-1])

# # -------------------------------
# # TEST ACCURACY
# # -------------------------------
# test_loss, test_acc = model.evaluate(X_test, Y_test)
# print("\nTest Accuracy:", test_acc)





# import numpy as np
# import os
# import cv2
# import tensorflow as tf
# from sklearn.utils import shuffle
# from keras.models import Sequential
# from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout
# import time


# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# # -------------------------------
# # PARAMETERS
# # -------------------------------
# image_size = 128
# labels = ['glioma', 'meningioma','notumor','pituitary']

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
# print("Loading data...")
# X_train, Y_train = load_data(train_path)
# X_test, Y_test = load_data(test_path)

# print("X_train shape:", X_train.shape)
# print("X_test shape:", X_test.shape)
# print("Sample pixel range:", X_train.min(), X_train.max())

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
# # EPOCH LIST
# # -------------------------------
# epoch_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

# results = []

# # -------------------------------
# # TRAIN LOOP
# # -------------------------------
# for ep in epoch_list:
#     print(f"\n===== Training with {ep} epochs =====\n")

#     # NEW MODEL (IMPORTANT)
#     model = Sequential()

#     model.add(Conv2D(32,(3,3),activation='relu',input_shape=(128,128,3)))
#     model.add(Conv2D(64,(3,3),activation='relu'))
#     model.add(MaxPooling2D(2,2))
#     model.add(Dropout(0.3))

#     model.add(Conv2D(64,(3,3),activation='relu'))
#     model.add(MaxPooling2D(2,2))
#     model.add(Dropout(0.3))

#     model.add(Conv2D(128,(3,3),activation='relu'))
#     model.add(Conv2D(128,(3,3),activation='relu'))
#     model.add(MaxPooling2D(2,2))
#     model.add(Dropout(0.3))
    
#     model.add(Conv2D(256,(3,3),activation='relu'))
#     model.add(MaxPooling2D(2,2))
#     model.add(Dropout(0.3))

#     model.add(Flatten())

#     model.add(Dense(512,activation='relu'))
#     model.add(Dropout(0.4))
#     model.add(Dense(128,activation='relu'))
#     # model.add(Dropout(0.3))

#     model.add(Dense(4,activation='softmax'))
   

#     # model.add(Conv2D(32,(3,3),activation='relu',input_shape=(128,128,3)))
#     # model.add(MaxPooling2D(2,2))

#     # model.add(Conv2D(64,(3,3),activation='relu'))
#     # model.add(MaxPooling2D(2,2))

#     # model.add(Conv2D(128,(3,3),activation='relu'))
#     # model.add(MaxPooling2D(2,2))

#     # model.add(Conv2D(256,(3,3),activation='relu'))
#     # model.add(MaxPooling2D(2,2))

#     # model.add(Flatten())
    
#     # model.add(Dense(256,activation='relu'))  
#     # model.add(Dropout(0.3))

#     # model.add(Dense(128,activation='relu'))

#     # model.add(Dense(4,activation='softmax'))

#     # -------------------------------
#     # COMPILE
#     # -------------------------------
#     model.compile(
#         loss='categorical_crossentropy',
#         optimizer='Adam',
#         metrics=['accuracy']
#     )

#     # -------------------------------
#     # TRAIN
#     # -------------------------------
#     start_time = time.time()

#     history = model.fit(
#         X_train, Y_train,
#         epochs=ep,
#         batch_size=16,
#         validation_data=0.2,
#         verbose=1
#     )

#     end_time = time.time()
#     training_time = end_time - start_time

#     print(f"Training time for {ep} epochs: {training_time:.2f} sec ({training_time/60:.2f} min)")

#     # -------------------------------
#     # FINAL ACCURACY
#     # -------------------------------
#     train_acc = history.history['accuracy'][-1]
#     val_acc = history.history['val_accuracy'][-1]

#     # -------------------------------
#     # TEST ACCURACY
#     # -------------------------------
#     test_loss, test_acc = model.evaluate(X_test, Y_test)

#     # -------------------------------
#     # SAVE MODEL
#     # -------------------------------
#     model.save(f"model_{ep}.h5")

#     # STORE RESULTS
#     results.append((ep, train_acc, val_acc, test_acc, training_time))

# # -------------------------------
# # PRINT RESULTS TABLE
# # -------------------------------
# print("\n================ FINAL RESULTS ================\n")
# print(f"{'Epochs':<10}{'Train Acc':<15}{'Val Acc':<15}{'Test Acc':<15}{'Time (sec)':<15}")

# for ep, train_acc, val_acc, test_acc, t in results:
#     print(f"{ep:<10}{train_acc:<15.4f}{val_acc:<15.4f}{test_acc:<15.4f}{t:<15.2f}")






# import numpy as np
# import os
# import cv2
# import tensorflow as tf
# from sklearn.utils import shuffle
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout
# import time
# from tensorflow.keras.utils import plot_model
# from tensorflow.keras import Input

# # Suppress warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

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
# # model = Sequential()

# # model.add(Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)))
# # model.add(Conv2D(64, (3,3), activation='relu'))
# # model.add(MaxPooling2D(2,2))
# # model.add(Dropout(0.3))

# # model.add(Conv2D(64, (3,3), activation='relu'))
# # model.add(MaxPooling2D(2,2))
# # model.add(Dropout(0.3))

# # model.add(Conv2D(128, (3,3), activation='relu'))
# # model.add(Conv2D(128, (3,3), activation='relu'))
# # model.add(MaxPooling2D(2,2))
# # model.add(Dropout(0.3))

# # model.add(Conv2D(128, (3,3), activation='relu'))
# # model.add(MaxPooling2D(2,2))
# # model.add(Dropout(0.3))

# # model.add(Flatten())

# # model.add(Dense(256, activation='relu'))
# # model.add(Dropout(0.4))
# # model.add(Dense(128, activation='relu'))

# # model.add(Dense(4, activation='softmax'))



# model = Sequential([
#     Input(shape=(128,128,3)),

#     # Block 1
#     Conv2D(32,(3,3),activation='relu'),
#     Conv2D(32,(3,3),activation='relu'),
#     MaxPooling2D(2,2),

#     # Block 2
#     Conv2D(64,(3,3),activation='relu'),
#     MaxPooling2D(2,2),
#     Conv2D(64,(3,3),activation='relu'),
#     Conv2D(64,(3,3),activation='relu'),
#     MaxPooling2D(2,2),

#     # Block 3
#     Conv2D(128,(3,3),activation='relu'),
#     Conv2D(128,(3,3),activation='relu'),
#     Conv2D(128,(3,3),activation='relu'),
#     MaxPooling2D(2,2),
#     Dropout(0.3),

#     # Block 4
#     Conv2D(256,(3,3),activation='relu'),
#     Conv2D(256,(3,3),activation='relu'),
#     MaxPooling2D(2,2),
#     Dropout(0.3),

#     Flatten(),

#     Dense(512,activation='relu'),
#     Dropout(0.4),
#     Dense(128,activation='relu'),

#     Dense(4,activation='softmax')
# ])

# # -------------------------------
# # SUMMARY
# # -------------------------------
# model.summary()

# # -------------------------------
# # SAVE MODEL DIAGRAM
# # -------------------------------
# plot_model(
#     model,
#     to_file='model.png',
#     show_shapes=True,
#     show_layer_names=True
# )

# print("Model diagram saved as model.png")


# # -------------------------------
# # COMPILE
# # -------------------------------
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
#     loss='categorical_crossentropy',
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
# model.save("model.h5")

# # -------------------------------
# # FINAL RESULTS
# # -------------------------------
# print("\nFinal Training Accuracy:", history.history['accuracy'][-1])
# print("Final Validation Accuracy:", history.history['val_accuracy'][-1])


import numpy as np
import os
import cv2
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout
import time
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Input

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# -------------------------------
# PARAMETERS
# -------------------------------
image_size = 128
labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

train_path = "dataset/Training"
test_path = "dataset/Testing"

# -------------------------------
# PREPROCESS FUNCTION (FIXED 🔥)
# -------------------------------
def preprocess(img):
    # Convert to grayscale (IMPORTANT FIX)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize
    img = cv2.resize(img, (image_size, image_size))

    # Noise reduction
    img = cv2.GaussianBlur(img, (3,3), 0)

    # Improve contrast (helps MRI)
    img = cv2.equalizeHist(img)

    # Normalize
    img = img.astype('float32') / 255.0

    # Convert back to 3 channels
    img = np.stack((img,)*3, axis=-1)

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
print("Pixel range:", X_train.min(), X_train.max())

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

# -------------------------------
# MODEL (UNCHANGED ARCHITECTURE ✔)
# # -------------------------------
# model = Sequential([
#     Input(shape=(128,128,3)),

#     # Block 1
#     Conv2D(32,(3,3),activation='relu'),
#     Conv2D(32,(3,3),activation='relu'),
#     MaxPooling2D(2,2),

#     # Block 2
#     Conv2D(64,(3,3),activation='relu'),
#     MaxPooling2D(2,2),
#     Conv2D(64,(3,3),activation='relu'),
#     Conv2D(64,(3,3),activation='relu'),
#     MaxPooling2D(2,2),

#     # Block 3
#     Conv2D(128,(3,3),activation='relu'),
#     Conv2D(128,(3,3),activation='relu'),
#     Conv2D(128,(3,3),activation='relu'),
#     MaxPooling2D(2,2),
#     Dropout(0.3),

#     # Block 4
#     Conv2D(256,(3,3),activation='relu'),
#     Conv2D(256,(3,3),activation='relu'),
#     Dropout(0.3),

#     Flatten(),

#     Dense(512,activation='relu'),
#     Dropout(0.4),
#     Dense(128,activation='relu'),

#     Dense(4,activation='softmax')
# ])

# # -------------------------------
# # SUMMARY
# # -------------------------------
# model.summary()


model = Sequential([
    Input(shape=(128,128,3)),

    # -------------------------------
    # Block 1
    # -------------------------------
    Conv2D(32, (3,3), activation='relu'),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    # -------------------------------
    # Block 2
    # -------------------------------
    Conv2D(64, (3,3), activation='relu'),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    # -------------------------------
    # Block 3
    # -------------------------------
    Conv2D(128, (3,3), activation='relu'),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.3),

    # -------------------------------
    # Block 4
    # -------------------------------
    Conv2D(256, (3,3), activation='relu'),
    Conv2D(256, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.3),

    # -------------------------------
    # Fully Connected
    # -------------------------------
    Flatten(),

    Dense(512, activation='relu'),
    Dropout(0.3),

    Dense(128, activation='relu'),

    Dense(4, activation='softmax')
])

model.summary()

# # -------------------------------
# # SAVE MODEL DIAGRAM
# # -------------------------------
# plot_model(
#     model,
#     to_file='model.png',
#     show_shapes=True,
#     show_layer_names=True
# )

# print("Model diagram saved as model.png")

# -------------------------------
# COMPILE
# -------------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------------------
# TRAIN
# -------------------------------
start_time = time.time()

history = model.fit(
    X_train, Y_train,
    epochs=25,
    batch_size=16,   # 🔥 safer than 16
    validation_split=0.2,
    verbose=1
)

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
model.save("model.keras")   # updated format

# -------------------------------
# FINAL RESULTS
# -------------------------------
print("\nFinal Training Accuracy:", history.history['accuracy'][-1])
print("Final Validation Accuracy:", history.history['val_accuracy'][-1])