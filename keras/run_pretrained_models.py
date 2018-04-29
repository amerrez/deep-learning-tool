# Common imports
import subprocess

import numpy as np
import os
import sys

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

# from models.research.object_detection.utils import ops as utils_ops
# from models.research.object_detection.utils import label_map_util
# from models.research.object_detection.utils import visualization_utils as vis_util



#Image to text constants
CHECKPOINT_PATH = 'im2text/kranti/Pretrained-Show-and-Tell-model/model_renamed.ckpt-2000000'
VOCAB_PATH = 'im2text/kranti/Pretrained-Show-and-Tell-model/word_counts.txt'

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

def load_object_detection_model():
    # What model to download.
    MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

    NUM_CLASSES = 90

    # ## Download Model

    # In[ ]:
    try:
        tar_file = tarfile.open(MODEL_FILE)
        print(' Object detection model\'s tar file already exits')
    except IOError:
        print('Downloading object detection model tar file')
        opener = urllib.request.URLopener()
        opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
        tar_file = tarfile.open(MODEL_FILE)
    finally:
        print('Loading object detection model')
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, os.getcwd())
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    print('Tensorflow graph for object detection model created')

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    print('Labels and categories for object detection loaded')


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict


if __name__=='__main__':
    #image_classification("sample_imgs/dog_sample.jpeg")
    img2text('sample_imgs/girl_riding_horse.jpeg')