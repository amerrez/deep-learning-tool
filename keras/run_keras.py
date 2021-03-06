import base64
import io
import json
import sys
import time
import os

import numpy as np
import redis
from threading import Thread
import zipfile

# import the necessary packages
from PIL import Image
from keras.applications import VGG16, imagenet_utils
from keras.preprocessing.image import img_to_array

# from run_pretrained_models import image_classification_server, img2text_server, object_detection

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANS = 3
IMAGE_DTYPE = "float32"

# initialize constants used for server queuing
IMAGE_QUEUE = "image_queue"
FACE_ID_QUEUE = "face_ID_queue"
BATCH_SIZE = 32
SERVER_SLEEP = 0.25
CLIENT_SLEEP = 0.25

# db = redis.StrictRedis(host="ec2-34-209-90-235.us-west-2.compute.amazonaws.com", port=6379, db=0)
db = redis.StrictRedis(host="localhost", port=6379, db=0)
model = None

print("* Loading model...")
model = VGG16(weights='imagenet', include_top=True)
print("* Model loaded")
PATH_TO_SAVE_UNZIPPED_FILES = '/Users/thuanbao/Study/249/test'


def base64_decode_image(image, dtype, shape):
    # if this is Python 3, we need the extra step of encoding the
    # serialized NumPy string as a byte object

    image = base64.b64decode(image)
    image = Image.open(io.BytesIO(image))
    # image = img_to_array(image)
    # image = np.expand_dims(image, axis=0)
    # image = imagenet_utils.preprocess_input(image)

    # if sys.version_info.major == 3:
    #     image = bytes(image, encoding="utf-8")

    # convert the string to a NumPy array using the supplied data
    # type and target shape
    # image = np.frombuffer(base64.decodestring(image), dtype=dtype)
    # image = image.reshape(shape)

    # return the decoded image
    return image


def convert_preds(preds):
    return list(map(lambda pred: {'label': pred[1], 'probability': str(pred[2])}, preds))


def classify_process():
    # continually pool for new images to classify
    while True:
        # attempt to grab a batch of images from the database, then
        # initialize the image IDs and batch of images themselves
        queue = db.lrange(IMAGE_QUEUE, 0, BATCH_SIZE - 1)

        # loop over the queue
        for q in queue:
            # deserialize the object and obtain the input image
            q = json.loads(q.decode("utf-8"))
            image = base64_decode_image(q["image"], IMAGE_DTYPE,
                                        (1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANS))
            type = q["type"]

            if type == 'IMC':
                preds = image_classification_server(image, model)
                db.set(q["id"], json.dumps(convert_preds(preds)))
            elif type == 'IMT':
                result = img2text_server(image)
                db.set(q["id"], result)
            elif type == 'ODT':
                res_img_byt_str = object_detection(image)
                db.set(q["id"], res_img_byt_str)
                # img = Image.open(io.BytesIO(res_img_byt_str))
                # img.show()

            # write back data to redis


            # remove the set of images from our queue
            db.ltrim(IMAGE_QUEUE, 1, -1)

            # sleep for a small amount
            time.sleep(SERVER_SLEEP)

def	face_id():
    # continually pool for new zip file of images to train
    while True:
        # attempt to grab a batch of images from the database, then
        # initialize the image IDs and batch of images themselves
        queue = db.lrange(FACE_ID_QUEUE, 0, BATCH_SIZE - 1)

        # loop over the queue
        for q in queue:
            # deserialize the object and obtain the input image
            q = json.loads(q.decode("utf-8"))
            zippedFile = base64.b64decode(q["file"])

            import zipfile
            zip_ref = zipfile.ZipFile(io.BytesIO(zippedFile), 'r')
            zip_ref.extractall(PATH_TO_SAVE_UNZIPPED_FILES)
            zip_ref.close()

            ##TODO: @bao call your script here

            # remove the set of images from our queue
            db.ltrim(FACE_ID_QUEUE, 1, -1)

            # sleep for a small amount
            time.sleep(SERVER_SLEEP)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    # load the function used to classify input images in a *separate*
    # thread than the one used for main classification
    # print("* Starting model service...")
    # t = Thread(target=face_id, args=())
    # t.daemon = True
    # t.start()
    # classify_process()
    face_id()
