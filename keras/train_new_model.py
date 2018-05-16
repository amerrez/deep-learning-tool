import numpy as np
from keras import applications
from keras import optimizers
from keras.applications.vgg16 import preprocess_input
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dropout, Flatten, Dense
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator


def train_face_id_model(directory):
    img_width, img_height = 512, 512
    train_data_dir = directory + "train"
    validation_data_dir = directory + "val"
    nb_train_samples = 500
    nb_validation_samples = 100
    batch_size = 16
    epochs = 10

    model = applications.VGG19(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3))

    # Freeze the not trained layers (first 12)
    for layer in model.layers[:10]:
        layer.trainable = False

    # Adding custom Layers
    x = model.output
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(3, activation="softmax")(x)

    # creating the final model 
    model_final = Model(input=model.input, output=predictions)

    # compile the model 
    model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
                        metrics=["accuracy"])

    # print out working progress
    print("Model started training....")

    # Initiate the train and test generators with data Augumentation 
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
        fill_mode="nearest",
        zoom_range=0.3,
        width_shift_range=0.3,
        height_shift_range=0.3,
        rotation_range=30)

    test_datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
        fill_mode="nearest",
        zoom_range=0.3,
        width_shift_range=0.3,
        height_shift_range=0.3,
        rotation_range=30)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode="categorical")

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        class_mode="categorical")

    # Save the model according to the conditions  
    checkpoint = ModelCheckpoint("transfer_keras_model.h5", monitor='val_acc', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='auto', period=1)
    early = EarlyStopping(monitor='val_acc', min_delta=0, patience=4, verbose=1, mode='auto')

    # Train the model
    model_final.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        epochs=epochs,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples,
        callbacks=[checkpoint, early])


def image_detection():
    model = load_model('transfer_keras_model.h5')
    img_path = '/Users/thuanbao/Downloads/test_data/amer.jpg'
    img = image.load_img(img_path, target_size=(256, 256))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    for (index, i), x in np.ndenumerate(preds):
        if i == 0:
            print("Amer probability is {} '<br>'".format(x))
        elif i == 1:
            print("Bao probability is {} '<br>'".format(x))
        else:
            print("Neither of Us probability is {}".format(x))
