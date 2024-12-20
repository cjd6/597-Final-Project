import numpy as np # linear algebra
import pandas as pd

import tensorflow as tf
import keras
import keras_cv
from keras import Sequential, Model
#from keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Flatten, Dropout, Activation, Input, Concatenate
#from keras.utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator
from keras import optimizers
#from keras.callbacks import ReduceLROnPlateau, EarlyStopping
#from keras.applications import efficientnet_v2, VGG16, ResNet50,InceptionResNetV2, InceptionV3

#from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

seed = 2022
np.random.seed(seed)
tf.random.set_seed(seed)

train_directory = "Data 2/Food_Grading2/train"
val_directory = "Data 2/Food_Grading2/val"
test_directory = "Data 2/Food_Grading2/test"

train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=15, width_shift_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
train_directory,
target_size=(256,256),
color_mode='rgb',
batch_size=32,
class_mode='binary',
subset='training',
shuffle=True,
seed=42
)

test_generator = test_datagen.flow_from_directory(
test_directory,
target_size=(256,256),
color_mode='rgb',
batch_size=32,
class_mode='binary',
shuffle=False
)

val_generator = test_datagen.flow_from_directory(
val_directory,
target_size=(256,256),
color_mode='rgb',
batch_size=32,
class_mode='binary',
shuffle=False
)

input_shape = (256, 256, 3) #Cifar10 image size
resized_shape = (224, 224, 3) #EfficientNetV2B0 model input shape
num_classes = 1

def build_model():
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Lambda(lambda image: tf.image.resize(image, resized_shape[:2]))(inputs) #Resize image to  size 224x224
    #base_model = keras.models.InceptionV3(include_top=False, input_shape=resized_shape, weights="imagenet")
    base_model = keras.applications.VGG16(include_top=False, input_shape=resized_shape, weights="imagenet")
    base_model.trainable = False
    x = base_model(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="sigmoid")(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

model = build_model()
#model.load_weights("VGGV2.keras")

plateau = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.7, patience=1, verbose=1)
earlystopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, verbose=1)

model.compile(optimizer=optimizers.SGD(learning_rate=0.003, momentum=0.9),
              loss="binary_crossentropy",
              metrics=["accuracy"])

model.summary()
print("\n")

history = model.fit(train_generator,
                    epochs=25,
                    validation_data=val_generator,
                    callbacks=[plateau, earlystopping, keras.callbacks.ModelCheckpoint(filepath="VGGAt{epoch}.keras")]
                   )

model.save("VGGV2.keras")

print("Evaluate on test data")
results = model.evaluate(test_generator, batch_size=32)
print("test loss, test acc:", results)

fig, ax = plt.subplots(figsize=(12, 4), nrows=1, ncols=2)
ax[0].plot(history.history["loss"], c="r", label="train loss")
ax[0].plot(history.history["val_loss"], c="b", label="val loss")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Loss")
ax[0].legend()

ax[1].plot(history.history["accuracy"], c="r", label="train accuracy")
ax[1].plot(history.history["val_accuracy"], c="b", label="val accuracy")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Accuracy")
ax[1].legend()