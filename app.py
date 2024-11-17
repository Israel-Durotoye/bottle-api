from bottle import Bottle, request, response, run
import tensorflow as tf
import numpy as np
from PIL import Image
import io

import gdown

# Replace with your file's shared link
url = "https://drive.google.com/uc?export=download&id=1fnQOvlcQ3hjs7SPJJrf8EnKiui_Ukthu"
output = "pest_classification_model.h5"
gdown.download(url, output, quiet=False)

# Initialize the Bottle app
app = Bottle()

# Load the pre-trained model
model = tf.keras.models.load_model("pest_classification_model.h5")

# Route for the image prediction
@app.post('/predict')
def predict():
    # Get the image from the request
    upload = request.files.get('file')  # 'file' is the key for file upload
    if not upload:
        return {"error": "No file uploaded"}

    try:
        # Read and process the image
        image = Image.open(io.BytesIO(upload.file.read()))
        image = image.convert("RGB")  # Ensure the image is in RGB mode
        image = image.resize((150, 150))  # Resize image to 150x150 as per model input size
        img_array = np.array(image) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict the class
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])  # Get the index of the highest prediction
        confidence = float(np.max(predictions[0]))  # Get the confidence of the prediction

        # Return the result
        return {"class": int(predicted_class), "confidence": confidence}

    except Exception as e:
        return {"error": str(e)}

# Run the Bottle app
run(app, host="0.0.0.0", port=10000) 

