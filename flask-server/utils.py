from io import StringIO
import base64

import numpy as np
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array


def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)

    buf = StringIO.StringIO()
    image.save(buf, format="PNG")
    image_string = buf.getvalue()

    # image = img_to_array(image)
    # image = np.expand_dims(image, axis=0)
    # image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return base64_encode_image(image_string)


def base64_encode_image(a):
    # base64 encode the input NumPy array
    return base64.b64encode(a).decode("utf-8")
