import os
from flask import Flask, request, jsonify
from flask_cors import *
import dill
import json
import traceback

BASE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
MODEL_DIR = '{}/models'.format(BASE_DIR)

with open('{}/best_senti_model.pkl'.format(MODEL_DIR), 'rb') as f:
    model = dill.load(f)

app = Flask(__name__)
CORS(app, supports_credentials=True)


@app.route('/api/textcnn/predict', methods=['POST'])
def textcnn_predict():

    try:
        # text needs to be a json array
        text = request.json['text']
        print(request.json)
        result = model.predict_prob(text)
        result = result.astype(float)
        result = list(result)
        result = [round(x, 2) for x in result]
        print(result)
        ret = {
            'status': 'success',
            'score': result,
        }
    except Exception as e:
        tb = traceback.format_exc()
        ret = {
            'status': 'error',
            'error_msg': str(tb),
        }
    return jsonify(ret)


# if __name__ == '__main__':
#     app.run(host='127.0.0.1', debug=True, port=5000)
