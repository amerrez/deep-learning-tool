# Common imports
import subprocess

import numpy as np
import os
import sys
import io

# Image classificaiton imports
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input,decode_predictions

# Image captioning imports
import pickle

# Object detection imports
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

#object detection import
sys.path.append("/Users/rahuldalal/Workspace/cs249/proj_2/dl_models/dl_models/models/research")
from object_detection.object_detection_tutorial import detect

#Image to text constants
CHECKPOINT_PATH = 'im2text/kranti/Pretrained-Show-and-Tell-model/model_renamed.ckpt-2000000'
VOCAB_PATH = 'im2text/kranti/Pretrained-Show-and-Tell-model/word_counts.txt'

#Cosntants for object detection
IMAGE_SIZE = (12,8)

#Run directly on image path
def image_classification(img_path):
    #Don't create the model here ...Create it outside
    model = VGG16(weights = 'imagenet', include_top = True)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    print('*****Image classification result ****** \n{}'.format(decode_predictions(preds, top=3)[0]))
    # print(type(decode_predictions(preds, top=3)[0]))
    return decode_predictions(preds, top=3)[0]

# Run from run_keras by passing image and model
def image_classification_server(img, model):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    print('*****Image classification result ****** \n{}'.format(decode_predictions(preds, top=3)[0]))
    # print(type(decode_predictions(preds, top=3)[0]))
    return decode_predictions(preds, top=3)[0]

#Run directly on image path
def img2text(img_path):
    #Get absolute paths
    checkpoint_path, vocab_path, img_path = os.path.abspath(CHECKPOINT_PATH), os.path.abspath(VOCAB_PATH), os.path.abspath(img_path)
    os.chdir('models/research/im2txt')
    os.system('bazel-bin/im2txt/run_inference \
    --checkpoint_path={} \
    --vocab_file={} \
    --input_files={}'.format(checkpoint_path, vocab_path, img_path))
    with open('img_capt_result', 'rb') as file:
        result = pickle.load(file)
    print('Result:{}'.format(result))
    return result

# Run from run_keras by passing image
def img2text_server(img):
    IMG_PATH = 'temp.jpeg'
    # im = Image.fromarray(img)
    img.save('temp.jpeg')
    #Get absolute paths
    checkpoint_path, vocab_path, img_path = os.path.abspath(CHECKPOINT_PATH), os.path.abspath(VOCAB_PATH), os.path.abspath(IMG_PATH)
    os.chdir('models/research/im2txt')
    # os.system('bazel-bin/im2txt/run_inference \
    # --checkpoint_path={} \
    # --vocab_file={} \
    # --input_files={}'.format(checkpoint_path, vocab_path, img_path))

    batcmd = 'bazel-bin/im2txt/run_inference \
        --checkpoint_path={} \
        --vocab_file={} \
        --input_files={}'.format(checkpoint_path, vocab_path, img_path)

    result = subprocess.check_output(batcmd, shell=True)
    os.chdir('../../..')

    print('Result:{}'.format(result))
    return result

def object_detection(img_path):
    # print(os.getcwd())
    os.chdir('models/research/object_detection')
    return detect(img_path)

#this method is implemented by Bao to train new model
def train_face_id_model(directory):
    img_width, img_height = 256, 256
    train_data_dir = directory+ "/train"
    validation_data_dir = directory+ "/val"
    nb_train_samples = 500
    nb_validation_samples = 100 
    batch_size = 16
    epochs = 10

    model = applications.VGG16(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))

    # Freeze the not trained layers (first 10)
    for layer in model.layers[:10]:
        layer.trainable = False

    #Adding custom Layers 
    x = model.output
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(3, activation="softmax")(x)

    # creating the final model 
    model_final = Model(input = model.input, output = predictions)

    # compile the model 
    model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

    # Initiate the train and test generators with data Augumentation 
    train_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,
    fill_mode = "nearest",
    zoom_range = 0.3,
    width_shift_range = 0.3,
    height_shift_range=0.3,
    rotation_range=30)

    test_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,
    fill_mode = "nearest",
    zoom_range = 0.3,
    width_shift_range = 0.3,
    height_shift_range=0.3,
    rotation_range=30)

    train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size = (img_height, img_width),
    batch_size = batch_size, 
    class_mode = "categorical")

    validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size = (img_height, img_width),
    class_mode = "categorical")

    # Save the model according to the conditions  
    checkpoint = ModelCheckpoint("transfer_keras_model.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    early = EarlyStopping(monitor='val_acc', min_delta=0, patience=4, verbose=1, mode='auto')


    # Train the model 
    model_final.fit_generator(
    train_generator,
    samples_per_epoch = nb_train_samples,
    epochs = epochs,
    validation_data = validation_generator,
    nb_val_samples = nb_validation_samples,
    callbacks = [checkpoint, early])

if __name__=='__main__':
    #image_classification("sample_imgs/dog_sample.jpeg")
    #img2text('sample_imgs/girl_riding_horse.jpeg')
    object_detection("test_images/image1.jpg")