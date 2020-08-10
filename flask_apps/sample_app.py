from flask import Flask
app = Flask(__name__)

@app.route('/sample') # URL to browse to for function underneath to be called (aka an endpoint)
def running():
    return 'Flask is running...'