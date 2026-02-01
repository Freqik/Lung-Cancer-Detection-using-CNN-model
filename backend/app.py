import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from training.model import LungCancerCNN
from utils.preprocessing import preprocess_image

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'lung_model.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LungCancerCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img_file = request.files['image']
    temp_path = "temp_image.png"
    img_file.save(temp_path)

    try:
        image_tensor = preprocess_image(temp_path).to(device)
        with torch.no_grad():
            output = model(image_tensor)
            prediction = (output.item() > 0.5)

        if prediction:
            result_message = "Analysis indicates a possibility of abnormal findings that may require medical attention."
        else:
            result_message = "No signs of abnormal findings were detected in the submitted scan."

        return jsonify({"prediction": result_message})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    app.run(debug=True)
