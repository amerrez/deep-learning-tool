from io import BytesIO
import base64


def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)

    buf = BytesIO()
    image.save(buf, format="PNG")
    image_string = buf.getvalue()

    # return the processed image
    return base64_encode_image(image_string)


def base64_encode_image(a):
    return base64.encodestring(a).decode("utf-8")
