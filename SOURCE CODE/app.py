from flask import Flask, render_template, request, url_for, redirect, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import google.generativeai as genai

app = Flask(__name__)

# Configure Gemini API Key
genai.configure(api_key="AIzaSyAgGCFMOJgkAVKpN7HDXhtD09fkXztUKII")  # Replace

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
    keras_model = load_model(r"D:\Skin-Cancer-Prediction-main\SOURCE CODE\model\skin.h5")  #  Your model path
except Exception as e:
    print(f"Error loading Keras model: {e}")
    keras_model = None

def get_gemini_info(condition, prompt_type="description"):
    """Fetches info from Gemini."""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')

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
    """Handles image submission, prediction, and redirects to results."""
    if request.method == 'POST':
        img = request.files.get('my_image')  # Use .get()
        if not img:
            return jsonify({'error': 'No image provided'}), 400  # Return JSON error

        img_folder = "static/uploads/"
        os.makedirs(img_folder, exist_ok=True)
        img_path = os.path.join("uploads", img.filename)
        img.save(os.path.join("static", img_path))

        prediction, confidence = predict_label(os.path.join("static", img_path))

        if prediction in ("Model loading error", "Prediction failed"):
            return jsonify({'error': prediction}), 500 # Return JSON error

        description = get_gemini_info(prediction, "description")
        symptoms = get_gemini_info(prediction, "symptoms")
        treatment = get_gemini_info(prediction, "treatment")

        #  Return a JSON response with a redirect URL
        return jsonify({
            'redirect': url_for('show_results',
                                prediction=prediction,
                                confidence=confidence,
                                img_path=img_path,
                                ai_description=description,
                                ai_symptoms=symptoms,
                                ai_treatment=treatment)
        })


@app.route("/success")  #  This route handles the *redirect*
def show_results():
    # Get all the necessary data from the *query parameters*
    prediction = request.args.get('prediction')
    confidence = request.args.get('confidence')
    img_path = request.args.get('img_path')
    ai_description = request.args.get('ai_description')
    ai_symptoms = request.args.get('ai_symptoms')
    ai_treatment = request.args.get('ai_treatment')

    if not all([prediction, confidence, img_path, ai_description, ai_symptoms, ai_treatment]):
        return "Error: Missing result data", 400 # Or redirect to an error page

    return render_template(
        "prediction.html",
        prediction=prediction,
        confidence=confidence,
        img_path=img_path,
        ai_description=ai_description,
        ai_symptoms=ai_symptoms,
        ai_treatment=ai_treatment,
    )

@app.route("/")
@app.route("/first")
def first():
    return render_template('first.html')


@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username == "admin" and password == "admin":
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error="Invalid Credentials!")
    else:
        return render_template('login.html', error=None)


@app.route("/index", methods=['GET', 'POST'])
def index():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)