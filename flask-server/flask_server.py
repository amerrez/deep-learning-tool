import io
import json
import uuid

import flask
import time

from PIL import Image
from flask import render_template, flash, redirect, request

# initialize our Flask application, Redis server, and Keras model
from werkzeug.utils import secure_filename

from config import Config
from forms import LoginForm
from utils import prepare_image, base64_encode_image

app = flask.Flask(__name__)
app.config.from_object(Config)

# db redis
import redis

db = redis.StrictRedis(host="localhost", port=6379, db=0)

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANS = 3
IMAGE_DTYPE = "float32"
SERVER_SLEEP = 0.25
CLIENT_SLEEP = 0.25
IMAGE_QUEUE = "image_queue"


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', title='Amer')


@app.route('/account')
def account():
    return render_template('account.html', title='Amer')


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
#         flash('Login requested for user {}, remember_me={}'.format(
#             form.username.data, form.remember_me.data))
        return redirect('/data_upload')
    return render_template('login.html', title='Sign In', form=form)


@app.route('/data_upload', methods=['GET', 'POST'])
def upload():
<<<<<<< Updated upstream
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
        return 'file uploaded successfully'

=======
	form = UploadForm()
	if form.validate_on_submit():
	#saving the file to the disk
		fileName = form.file.data.filename
		form.file.data.save('uploads/' + fileName)

# 		sending it to redis would be somethink likke this
# 			k = str(uuid.uuid4())
# 			d = {"id": k, "image": base64_encode_image(image)}
# 			db.rpush(IMAGE_QUEUE, json.dumps(d))		

		return redirect('/index')
	return render_template('data_upload.html', title='Upload', form=form)

@app.route('/train')	
def simple_train():
	# USAGE
	# python simple_request.py

	# import the necessary packages
	import requests

	# initialize the Keras REST API endpoint URL along with the input
	# image path
	KERAS_REST_API_URL = "http://localhost:5000/predict"
# 	return '''<p> link http://localhost:5000/predict loaded</p>'''
	IMAGE_PATH = "jemma.png"
# 	return '''<p> image path loaded</p>'''
	# load the input image and construct the payload for the request
	image = open(IMAGE_PATH, "rb").read()
	payload = {"image": image}
# 	return '''<p> image opened and loaded</p>'''
	# submit the request
	r = requests.post(KERAS_REST_API_URL, files=payload).json()
	
	return '''<p> request Json returned to variable r</p>'''
	# ensure the request was sucessful
	if r["success"]:
		return '''<p> request suceeded</p>'''
>>>>>>> Stashed changes

@app.route('/train')
def simple_train():
    # USAGE
    # python simple_request.py

    # import the necessary packages
    import requests

    # initialize the Keras REST API endpoint URL along with the input
    # image path
    KERAS_REST_API_URL = "http://localhost:5000/predict"
    # 	return '''<p> link http://localhost:5000/predict loaded</p>'''
    IMAGE_PATH = "jemma.png"
    # 	return '''<p> image path loaded</p>'''
    # load the input image and construct the payload for the request
    image = open(IMAGE_PATH, "rb").read()
    payload = {"image": image}
    # 	return '''<p> image opened and loaded</p>'''
    # submit the request
    r = requests.post(KERAS_REST_API_URL, files=payload).json()

    return '''<p> request Json returned to variable r</p>'''
    # # ensure the request was sucessful
    # if r["success"]:
    #     return '''<p> request suceeded</p>'''
    #
    #     # loop over the predictions and display them
    #     return render_tempalte('index.html', task='show_results', data=r["predictions"])
    # # 		for (i, result) in enumerate(r["predictions"]):
    # # 			print("{}. {}: {:.4f}".format(i + 1, result["label"],
    # # 				result["probability"]))
    #
    # # otherwise, the request failed
    # else:
    #     return '''<p> request failed</p>'''
    #     print("Request failed")

	# otherwise, the request failed
	else:
		return '''<p> request failed</p>'''	
		print("Request failed")	
	

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format and prepare it for
            # classification
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            image = prepare_image(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
            
            #reading the task choice
            task = flask.request.form["task"]
            			
            # ensure our NumPy array is C-contiguous as well,
            # otherwise we won't be able to serialize it
            image = image.copy(order="C")

            # generate an ID for the classification then add the
            # classification ID + image to the queue
            k = str(uuid.uuid4())
            d = {"id": k, "image": base64_encode_image(image), "type":task}
            db.rpush(IMAGE_QUEUE, json.dumps(d))

            # keep looping until our model server returns the output
            # predictions
            while True:
                # attempt to grab the output predictions
                output = db.get(k)

                # check to see if our model has classified the input
                # image
                if output is not None:
                    # add the output predictions to our data
                    # dictionary so we can return it to the client
                    output = output.decode("utf-8")
                    data["predictions"] = json.loads(output)

                    # delete the result from the database and break
                    # from the polling loop
                    db.delete(k)
                    break

                # sleep for a small amount to give the model a chance
                # to classify the input image
                time.sleep(CLIENT_SLEEP)

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


# if this is the main thread of execution
# then start the server
if __name__ == "__main__":
    # start the web server
    print("* Starting web service...")
    app.run()
