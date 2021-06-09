from flask import Flask, render_template
import random
import json
import base64
from recognition_camera import predict_expression, get_attention_score, get_user_photo
app = Flask(__name__)
capture, exp_model = predict_expression()

NAME='Yudong'

@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route('/data')
def get_data():
    return json.dumps({'name':NAME,'value':get_attention_score(capture, exp_model)})

@app.route('/camera')
def get_cam():
    return json.dumps({'name':NAME,'value':base64.b64encode(get_user_photo(capture)).decode('utf-8')})

if __name__ == '__main__':
    app.run()