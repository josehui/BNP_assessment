import News_Analysis

from flask import Flask, request, jsonify
from flask_cors import CORS

import json

app = Flask(__name__)
CORS(app) #to allow CORS for all domains on all routes, source: https://flask-cors.readthedocs.io/en/latest/

@app.route('/message/', methods=['GET','POST'])
def hello_world(name=None):
    msg = request.args.get('message')
    print (msg)
    str = {'key':'Hello World!'}
    #out = {'key':str}
    return jsonify(str)

if __name__ == '__main__':
    
    app.run(port=8080)