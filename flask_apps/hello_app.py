from flask import request
from flask import jsonify
from flask import Flask

app = Flask(__name__)  # create instance of Flask class


@app.route('/hello', methods=['POST'])  # 'POST' specifies the allowed HTTP request for this endpoint
# in this case, data will be sent in HTTP POST request
def hello():  # what to do when post request is received
    message = request.get_json(force=True)  # get message from client in json format
    name = message['name']  # extract the name from the message json with key 'name'
    response = {  # response to send back to web app
        'greeting': 'Hello, ' + name + '!'
    }
    return jsonify(response)  # convert python dictionary to json and return it
