import os
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Load model and class indices
MODEL_PATH = 'models/leaf_classifier.h5'
CLASS_INDICES_PATH = 'class_indices.json'

model = load_model(MODEL_PATH)

import json
with open(CLASS_INDICES_PATH, 'r') as f:
    class_indices = json.load(f)

# Reverse mapping from index to class name
classes = {v: k for k, v in class_indices.items()}

# === Disease info mapping ===
disease_info = {
    # Tomato diseases
    "Tomato__Target_Spot": {
        "name": "Target Spot",
        "solution": (
            "Remove all infected leaves and plant debris to prevent further spread. "
            "Apply fungicides such as chlorothalonil or copper-based sprays every 7–10 days. "
            "Ensure proper spacing between plants for airflow and avoid wetting leaves during irrigation."
        )
    },
    "Tomato__Tomato_mosaic_virus": {
        "name": "Tomato Mosaic Virus",
        "solution": (
            "Immediately remove and destroy infected plants to stop virus spread. "
            "Disinfect garden tools regularly with a bleach solution. "
            "Plant resistant tomato varieties and avoid smoking near plants as the virus can spread via hands."
        )
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "name": "Yellow Leaf Curl Virus",
        "solution": (
            "Control whiteflies using yellow sticky traps or insecticidal soap. "
            "Remove infected plants promptly. "
            "Use virus-resistant tomato varieties and practice crop rotation to reduce chances of infection."
        )
    },
    "Tomato_Bacterial_spot": {
        "name": "Bacterial Spot",
        "solution": (
            "Use only certified disease-free seeds and seedlings. "
            "Apply copper-based bactericides weekly during warm, wet conditions. "
            "Avoid overhead watering and maintain good spacing to reduce humidity around plants."
        )
    },
    "Tomato_Early_blight": {
        "name": "Early Blight",
        "solution": (
            "Remove infected leaves as soon as they appear. "
            "Spray fungicides containing mancozeb or chlorothalonil. "
            "Water at the base of the plants and rotate crops every 2–3 years."
        )
    },
    "Tomato_healthy": {
        "name": "Healthy",
        "solution": (
            "No action needed. Keep monitoring the plants regularly for any early signs of disease. "
            "Maintain proper watering and nutrition to keep plants strong."
        )
    },
    "Tomato_Late_blight": {
        "name": "Late Blight",
        "solution": (
            "Quickly remove and destroy infected plants to prevent spread. "
            "Apply fungicides containing copper or chlorothalonil at the first sign of symptoms. "
            "Avoid working in the garden when plants are wet."
        )
    },
    "Tomato_Leaf_Mold": {
        "name": "Leaf Mold",
        "solution": (
            "Improve air circulation by pruning lower leaves. "
            "Avoid overhead irrigation. "
            "If severe, apply fungicides containing mancozeb or copper at regular intervals."
        )
    },
    "Tomato_Septoria_leaf_spot": {
        "name": "Septoria Leaf Spot",
        "solution": (
            "Remove and destroy infected leaves immediately. "
            "Apply fungicides such as chlorothalonil every 7–10 days. "
            "Keep foliage dry and rotate crops yearly."
        )
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "name": "Spider Mites",
        "solution": (
            "Spray with neem oil or insecticidal soap every 5–7 days until infestation is under control. "
            "Increase humidity around plants and encourage beneficial insects such as ladybugs."
        )
    },

    # Chili leaf diseases
    "Bacterial-spot": {
        "name": "Bacterial Spot (Chili)",
        "solution": (
            "Remove infected leaves and avoid overhead watering. "
            "Apply copper-based bactericides weekly during humid conditions. "
            "Rotate crops and avoid planting chili in the same soil for at least 2 years."
        )
    },
    "Cercospora-leaf-spot": {
        "name": "Cercospora Leaf Spot",
        "solution": (
            "Prune infected leaves and improve airflow between plants. "
            "Spray fungicides containing copper or mancozeb every 10–14 days. "
            "Remove plant debris after harvest to prevent re-infection."
        )
    },
    "curl-virus": {
        "name": "Chili Curl Virus",
        "solution": (
            "Control whiteflies with sticky traps or neem oil sprays. "
            "Immediately remove infected plants to limit spread. "
            "Plant virus-resistant chili varieties when available."
        )
    },
    "Healthy-Leaf": {
        "name": "Healthy Leaf",
        "solution": (
            "No action needed. Continue providing proper watering, balanced fertilization, "
            "and regular observation to keep the plant in optimal health."
        )
    },
    "Nutrition-deficiency": {
        "name": "Nutrient Deficiency",
        "solution": (
            "Identify the specific nutrient lacking (nitrogen, potassium, magnesium, etc.) through soil testing. "
            "Apply the recommended fertilizer in balanced amounts. "
            "Mulch around plants to retain soil nutrients."
        )
    },
    "Unlabeled": {
        "name": "Unlabeled Data",
        "solution": (
            "The image may not match any trained category. "
            "Try uploading a clearer image, or update the training dataset with more labeled samples."
        )
    },
    "White-spot": {
        "name": "White Spot",
        "solution": (
            "Remove affected leaves and avoid excessive moisture on foliage. "
            "Apply sulfur-based fungicides every 10 days until symptoms subside. "
            "Maintain proper plant spacing to allow airflow."
        )
    }
}

# Prediction function
def model_predict(img_path, model):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))  # Ensure this matches your training size
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0  # Normalization

    preds = model.predict(x)
    pred_class = np.argmax(preds, axis=1)[0]
    class_name = classes[pred_class]

    return class_name

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', prediction="No file uploaded.")

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', prediction="No file selected.")

        if file:
            filepath = os.path.join('static', file.filename)
            file.save(filepath)

            pred_class_name = model_predict(filepath, model)
            plant_name = disease_info.get(pred_class_name, {}).get("name", "Unknown")
            solution = disease_info.get(pred_class_name, {}).get("solution", "No solution available.")

            return render_template('index.html',
                                   prediction=plant_name,
                                   solution=solution,
                                   image_url=filepath)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
