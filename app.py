from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model and labels
model = load_model("model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

@app.route("/")
def home():
    return "Dust Detection API is running"

# DEBUG ENDPOINT - Accepts raw image data
@app.route("/predict_debug", methods=["POST"])
def predict_debug():
    try:
        print(f"DEBUG: Content-Type: {request.content_type}")
        print(f"DEBUG: Content-Length: {request.content_length}")
        
        if request.content_type and 'multipart/form-data' in request.content_type:
            print("DEBUG: Multipart form-data detected")
            if 'image' not in request.files:
                return jsonify({"error": "No 'image' file in form-data", "content_type": request.content_type}), 400
            image_file = request.files['image']
            print(f"DEBUG: Filename: {image_file.filename}")
            image = Image.open(image_file.stream).convert("RGB")
        elif request.content_type and 'image/jpeg' in request.content_type:
            print("DEBUG: Raw JPEG detected")
            image_data = request.get_data()
            print(f"DEBUG: Raw data size: {len(image_data)}")
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        else:
            return jsonify({"error": f"Unsupported content type: {request.content_type}"}), 400
        
        # Process image
        image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array
        
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence = float(prediction[0][index])
        
        return jsonify({
            "class": class_name,
            "confidence": confidence,
            "debug": "success"
        })
    except Exception as e:
        print(f"DEBUG Error: {str(e)}")
        return jsonify({"error": str(e)}), 400

# Your original endpoint
@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image received"}), 400

    image = Image.open(request.files['image']).convert("RGB")
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence = float(prediction[0][index])

    return jsonify({
        "class": class_name,
        "confidence": confidence
    })

if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5000, debug=True)
