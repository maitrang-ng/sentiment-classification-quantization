import os
import onnx

from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForSequenceClassification
from predictor import torch_predict, onnx_predict
from constant import onnx_model_path, onnx_tokenizer, torch_model_path

# TODO 1
# Khởi tạo app flask
app = Flask(__name__)


# TODO 2: Tạo route /api/v1/model_info
# Khi client gọi GET. Trả về acc và size của model
@app.route('/api/v1/model_info', methods=['GET'])
def get_model_info_v1():
    model = AutoModelForSequenceClassification.from_pretrained(torch_model_path)
    model_size =  os.path.getsize("".join([torch_model_path, '/pytorch_model.bin']))/(1024*1024)
    result_dict = {
        "accuracy": 93.05,
        "size": model_size}
    if model:
        return jsonify(result_dict)
    else:
        return 'Failed to load model.'

@app.route('/api/v2/model_info', methods=['GET'])
def get_model_info_v2():
    model = onnx.load(onnx_model_path)
    model_size =  os.path.getsize(onnx_model_path)/(1024*1024)
    result_dict = {
        "accuracy": 92.05,
        "size": model_size}
    if model:
        return jsonify(result_dict)
    else:
        return 'Failed to load model.'


# TODO 3: Tạo route /api/v1/predict
# Khi client gọi POST, lay noi dung { "review": "xxx" } tu request body va tra ve ket qua prediction sentiment va score  
@app.route('/api/v1/predict', methods=['POST'])
def torch_process():
    # get data from request
    data = request.get_json()
    review = data.get('review', '')
    if not review:
        return jsonify({ "error": "review text is not provided" }), 400
    # get prediction
    result = torch_predict(torch_model_path, review)
    return jsonify(result)


# TODO 4: Tạo route /api/v2/predict
# Khi client gọi POST, lay noi dung { "review": "xxx" } tu request body va tra ve ket qua prediction sentiment va score  
@app.route('/api/v2/predict', methods=['POST'])
def onnx_process():
    # get data from request
    data = request.get_json()
    review = data.get('review', '')
    if not review:
        return jsonify({ "error": "review text is not provided" }), 400
    # get prediction
    result = onnx_predict(onnx_model_path, onnx_tokenizer, review)
    return jsonify(result)


# TODO 5: Khởi tạo ứng dụng ở cổng 80
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)

