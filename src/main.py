from flask import Flask, render_template, jsonify
from flask_cors import cross_origin, CORS

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
    return jsonify({
        'name':NAME,
        # 'value':random.random()
        'value': get_attention_score(capture, exp_model)
    })

@app.route('/camera')
@cross_origin()
def get_cam():
    return jsonify({'name':NAME,'value':base64.b64encode(get_user_photo(capture)).decode('utf-8')})

if __name__ == '__main__':
    app.run()