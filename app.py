from flask import Flask, render_template, request, jsonify
from flask_cors import CORS  # Import CORS from flask_cors module
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import base64
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes in your Flask app

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

# Define a route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image from the POST request
        img_data = request.form['image']
        img_data = base64.b64decode(img_data)

        # Convert the image data to a NumPy array
        img_array = image.img_to_array(image.load_img(io.BytesIO(img_data), target_size=(224, 224)))

        # Preprocess the image for the ResNet50 model
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Get model predictions
        predictions = model.predict(img_array)
        decoded_predictions = decode_predictions(predictions, top=3)[0]

        # Return predictions as JSON
        result = [{'label': label, 'probability': float(prob)} for (_, label, prob) in decoded_predictions]
        return jsonify(result)

    except Exception as e:
        print("Error:", str(e))
        return jsonify(error=str(e)), 500

if __name__ == '__main__':
    app.run(debug=True)
