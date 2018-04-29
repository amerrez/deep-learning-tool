from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input,decode_predictions
import numpy as np
import os
import sys

#Image to text constants
CHECKPOINT_PATH = '/Users/rahuldalal/Workspace/cs249/proj_2/dl_models/im2text/kranti/Pretrained-Show-and-Tell-model/model_renamed.ckpt-2000000'
VOCAB_PATH = '/Users/rahuldalal/Workspace/cs249/proj_2/dl_models/im2text/kranti/Pretrained-Show-and-Tell-model/word_counts.txt'


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

def img2text(img_path):
    #Get absolute paths
    checkpoint_path, vocab_path, img_path = os.path.abspath(CHECKPOINT_PATH), os.path.abspath(VOCAB_PATH), os.path.abspath(img_path)
    os.chdir('models/research/im2txt')
    os.system('bazel_bin/im2txt/run_inference \
    --checkpoint_path={} \
    --vocab_file={} \
    --input_files={}'.format(checkpoint_path, vocab_path, img_path))

if __name__=='__main__':
    #image_classification("sample_imgs/dog_sample.jpeg")
    img2text('sample_imgs/girl_riding_horse.jpeg')