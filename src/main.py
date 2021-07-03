from flask import Flask, jsonify
from flask_cors import cross_origin, CORS
import argparse
import sys

import json
import random
import base64
from recognition_camera import predict_expression, get_attention_score, get_user_photo
app = Flask(__name__)
# CORS(app, supports_credentials=True)
capture, exp_model = predict_expression()

NAME='Yudong'

@app.route('/')
def hello_world():
    return "Hello World!" #render_template("index.html")

@app.route('/data')
@cross_origin()
def get_data():
    global args
    return jsonify({
        'name' : NAME,
        'value': random.random() if args.useRand == 1 else get_attention_score(capture, exp_model)
    })

@app.route('/camera')
@cross_origin()
def get_cam():
    return jsonify({
        'name':NAME,
        'value':base64.b64encode(get_user_photo(capture)).decode('utf-8')
    })

if __name__ == '__main__':
    global args
    parser = argparse.ArgumentParser(description='simple flask backend')
    parser.add_argument('--use-rand', type=int, help='use random value (0/1)', default=0, dest="useRand")
    args = parser.parse_args()
    print(args)
    app.run(
        host='0.0.0.0',
        port=5000,
        #ssl_context=('cert.pem', 'key.pem')
    )
    #from waitress import serve
    #serve(app, host='0.0.0.0', port=5000)
