# deep-learning-tool

## Steps to run:
1. After cloning the repository, create a new virtual environment:
* Follow https://virtualenv.pypa.io/en/stable/ to install virtualevn
* Then from the root directory of the repository, run  `virtualenv venv` to create a new virtual environment
2. Follow https://redis.io/download to install redis. If you are using Mac, i recommend run `brew install redis` from your terminal. Then start redis server by running `redis-server`
3. Run `source ./venv/bin/activate` to activate your virtual environment
4. Run `pip install -r requirements.txt` install all dependencies
5. Run flask-server: `python ./flask-server/flask_server.py`. this will start a web server listening on port 5000. Go to http://local.staging.axlehire.com:5000/ to check if the server is up
6. Run Keras: open a new terminal and change dir to the repository's root folder, then activate virtual environment again: `source ./venv/bin/activate`. Run `python ./keras/run_keras_server.py` to kick off keras script 