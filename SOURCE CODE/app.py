from flask import Flask, render_template, request, url_for
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import google.generativeai as genai

app = Flask(__name__)

# Configure Gemini API Key
genai.configure(api_key="AIzaSyAgGCFMOJgkAVKpN7HDXhtD09fkXztUKII")  # Replace with YOUR API key!

# Class labels
verbose_name = {
    0: 'Actinic keratoses and intraepithelial carcinomae',
    1: 'Basal cell carcinoma',
    2: 'Benign keratosis-like lesions',
    3: 'Dermatofibroma',
    4: 'Melanocytic nevi',
    5: 'Pyogenic granulomas and hemorrhage',
    6: 'Melanoma',
}

# Load the Keras model
try:
    keras_model = load_model(r"D:\Skin-Cancer-Prediction-main\SOURCE CODE\model\skin.h5")
except Exception as e:
    print(f"Error loading Keras model: {e}")
    keras_model = None

def get_gemini_info(condition, prompt_type="description"):
    """Fetches info from Gemini, formatting treatment as a prescription."""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')  # Use gemini-pro

        if prompt_type == "description":
            prompt = f"Provide a concise medical description of {condition}."
        elif prompt_type == "symptoms":
            prompt = f"List the common symptoms of {condition} concisely."
        elif prompt_type == "treatment":
            prompt = (
                f"Generate a concise treatment outline for {condition} in a prescription-like format. "
                f"Include potential medication classes (no brand names) and lifestyle recommendations. "
                f"Use clear headings and make key information (like medication classes) bold. "
                f"Limit the response to 10 lines.  Do not include a disclaimer."
            )
        else:
            return "Invalid prompt type."

        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return f"Could not fetch information: {e}"


def predict_label(img_path):
    """Predicts skin condition and confidence."""
    if keras_model is None:
        return "Model loading error", 0.0

    try:
        test_image = image.load_img(img_path, target_size=(28, 28))
        test_image = image.img_to_array(test_image) / 255.0
        test_image = test_image.reshape(1, 28, 28, 3)
        predictions = keras_model.predict(test_image)
        class_index = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions) * 100
        return verbose_name[class_index], round(confidence, 2)
    except Exception as e:
        print(f"Prediction Error: {e}")
        return "Prediction failed", 0.0

@app.route("/submit", methods=['POST'])
def get_output():
    """Handles image submission and result display."""
    if request.method == 'POST':
        img = request.files['my_image']
        if not img:
            return render_template("index.html", error="No image selected.")

        img_folder = "static/uploads/"  # Corrected path
        os.makedirs(img_folder, exist_ok=True)
        img_path = os.path.join("uploads", img.filename)  # Corrected path
        img.save(os.path.join("static", img_path)) # Corrected: save to static/uploads

        prediction, confidence = predict_label(os.path.join("static", img_path)) # Use static path

        if prediction == "Model loading error" or prediction == "Prediction failed":
            return render_template("index.html", error=prediction)

        description = get_gemini_info(prediction, "description")
        symptoms = get_gemini_info(prediction, "symptoms")
        treatment = get_gemini_info(prediction, "treatment")
        # Corrected: use url_for with the *relative* path within static
        return render_template(
            "prediction.html",
            prediction=prediction,
            confidence=confidence,
            img_path=img_path,  # Pass the relative path
            ai_description=description,
            ai_symptoms=symptoms,
            ai_treatment=treatment,
        )

@app.route("/")
@app.route("/first")
def first():
    return render_template('first.html')

@app.route("/login")
def login():
    return render_template('login.html')

@app.route("/index", methods=['GET', 'POST'])
def index():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)