import numpy as np # linear algebra
import pandas as pd
import random

import tensorflow as tf
import keras
import keras_cv
from keras import Sequential, Model
from keras_preprocessing.image import ImageDataGenerator
from keras import optimizers

test_directory = "Data 2/Food_Grading2/test"
input_shape = (256, 256, 3) #Cifar10 image size
resized_shape = (224, 224, 3) #EfficientNetV2B0 model input shape
num_classes = 1

test_datagen = ImageDataGenerator(
    rescale=1.0 / 224,
    fill_mode='nearest'
)

test_generator = test_datagen.flow_from_directory(
    test_directory,
    target_size=(256, 256),
    batch_size=1,
    class_mode='binary',
    shuffle=False
)

# Quadratic confidence; 0.5 = 0 confidence, 0 or 1 = 1 confidence, quadratic scaling inbetween
def getConfidence(prediction):
    return 4 * (prediction - 0.5) ** 2

# First Past the Post Voting; all models' votes are counted equally regardless of confidence or competence
def FPTPVoting(models, trueLabels):
    totalRight = 0
    for i in range(len(models[0].predictions)):
        total = 0
        for model in models:
            pred = model.predictions[i]
            if pred >= 0.5:
                total += 1
            else:
                total -= 1
        overallPrediction = total >= 0 # 0 if majority of models say 0, 1 if majority say 1
        totalRight += overallPrediction == trueLabels[i]
    return totalRight / len(trueLabels)

# Confidence Voting; how confident models are in their predictions are taken into account
def confidenceVoting(models, trueLabels):
    totalRight = 0
    for i in range(len(models[0].predictions)):
        total = 0
        for model in models:
            pred = model.predictions[i]
            conf = getConfidence(pred)
            if pred >= 0.5:
                total += conf # If the model predicts 1, add confidence to total
            else:
                total -= conf # If the model predicts 0, subtract confidence from total

        overallPrediction = int(total >= 0) # 0 if 0-voting models were more confident, 1 if 1-voting models were
        totalRight += overallPrediction == trueLabels[i]
    return totalRight / len(test_generator)

# Competence Voting; how well models performed on test data set are taken into account
def competenceVoting(models, trueLabels):
    totalRight = 0
    for i in range(len(models[0].predictions)):
        total = 0
        for model in models:
            pred = model.predictions[i]
            if pred >= 0.5:
                total += model.weight # If the model predicts 1, add model's weight to total
            else:
                total -= model.weight # If the model predicts 0, subtract model's weight from total

        overallPrediction = int(total >= 0) # 0 if 0-voting models were more competent, 1 if 1-voting models were
        totalRight += overallPrediction == trueLabels[i]
    return totalRight / len(test_generator)

# Weighted Voting; how confident models are in their predictions, as well as their confidences are taken into account
def weightedVoting(models, trueLabels):
    totalRight = 0
    for i in range(len(models[0].predictions)):
        total = 0
        for model in models:
            pred = model.predictions[i]
            conf = getConfidence(pred)
            if pred >= 0.5:
                total += conf * model.weight # If the model predicts 1, add confidence times weight to total
            else:
                total -= conf * model.weight # If the model predicts 0, subtract confidence times weight from total

        overallPrediction = int(total >= 0) # 0 if 0-voting models were more confident, 1 if 1-voting models were
        totalRight += overallPrediction == trueLabels[i]
    return totalRight / len(test_generator)

def buildVGG():
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Lambda(lambda image: tf.image.resize(image, resized_shape[:2]))(inputs) #Resize image to  size 224x224
    base_model = keras.applications.VGG16(include_top=False, input_shape=resized_shape, weights="imagenet")
    base_model.trainable = False
    x = base_model(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="sigmoid")(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def buildEfficientnetV2S():
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Lambda(lambda image: tf.image.resize(image, resized_shape[:2]))(inputs) #Resize image to  size 224x224
    base_model = keras.applications.efficientnet_v2.EfficientNetV2S(include_top=False, input_shape=resized_shape, weights="imagenet")
    base_model.trainable = False
    x = base_model(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="sigmoid")(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def buildInceptionV3():
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Lambda(lambda image: tf.image.resize(image, resized_shape[:2]))(inputs) #Resize image to  size 224x224
    base_model = keras.applications.InceptionV3(include_top=False, input_shape=resized_shape, weights="imagenet")
    base_model.trainable = False
    x = base_model(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="sigmoid")(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

class trainedModel:
    def __init__(self, inputName, inputDir, inputValAcc, inputModel):
        self.name = inputName
        self.directory = inputDir
        self.weight = inputValAcc - 0.5 # How much weight to be given to the model, based on test accuracy
        #self.weight = (inputValAcc - 0.5) ** 2 # How much weight to be given to the model, based on test accuracy
        self.model = inputModel
        self.model.compile(loss="binary_crossentropy", metrics=["accuracy"])
        self.model.load_weights(inputDir)

    def print(self, printStructure = False):
        print("Model name:", self.name)
        print("Model directory:", self.directory)
        print("Model weight:", self.weight)
        if printStructure:
            print("Model structure:")
            self.model.summary()

    def predict(self, img, inputVerbose = 0):
        return (self.model.predict(img, verbose = inputVerbose))

# Models that always output a specific prediction for any input, with nudged confidence; used for debugging/testing
class testModel:
    def __init__(self, inputName, inputValAcc, inputMode, inputUncertainty):
        self.name = inputName
        self.weight = (inputValAcc - 0.5) ** 2 # How much weight to be given to the model, based on test accuracy
        self.mode = inputMode
        self.uncertainty = inputUncertainty

    def print(self):
        print("Model name:", self.name)
        print("Model mode:", self.mode)
        print("Model uncertainty:", self.uncertainty)
        print("Model weight:", self.weight)

    def predict(self, img):
        if self.mode == 0:
            res = min(random.random() * self.uncertainty, 0.4999)
        else:
            res = max(1 - (random.random() * self.uncertainty), 0.5)
        return [[res]]

# For testing; ignore
#testModels = [testModel("0 competent confident", 0.9, 0, 0.1), testModel("0 competent confident #2", 0.9, 0, 0.1),
#              testModel("1 competent confident", 0.9, 1, 0.1), testModel("1 competent confident #2", 0.9, 1, 0.1)]

# Name, directory of the weights, accuracy on test data set, base structure of the network
VGG = trainedModel("VGG", "Models/VGGFinal.keras", 0.90, buildVGG())
InceptionNet = trainedModel("InceptionNet", "Models/Inceptionv3_Model.keras", 0.97, buildInceptionV3())
EfficientNet = trainedModel("EfficientNet", "Models/Efficientnet_V2S_Model.keras", 0.67, buildEfficientnetV2S())

models = [VGG, InceptionNet, EfficientNet]

combos = [[VGG, InceptionNet, EfficientNet], [VGG, InceptionNet], [VGG, EfficientNet], [InceptionNet, EfficientNet]]

for model in models:
    #model.model.evaluate(test_generator)
    res = model.predict(test_generator, 1)
    preds = []
    for i in res:
        preds.append(i[0])
    model.predictions = preds

i = 1
for combo in combos:
    print("\nTesting combo", i)
    print("First past the post voting test accuracy:", FPTPVoting(combo, test_generator.labels))
    print("Confidence voting test accuracy:", confidenceVoting(combo, test_generator.labels))
    print("Competence voting test accuracy:", competenceVoting(combo, test_generator.labels))
    print("Weighted voting test accuracy:", weightedVoting(combo, test_generator.labels))
    i += 1

trueLabels = test_generator.labels

masks = []

for i in range(len(trueLabels)):
    correctMask = 0
    for j in range(3):
        if round(models[j].predictions[i], 0) == trueLabels[i]:
            correctMask += 2 ** j
    masks.append(correctMask)

# Odd = VGG got it right
# % 2 is odd = Inception got it right
# % 4 is odd = EfficientNet got it right

for i in range(8):
    print(i, masks.count(i))
