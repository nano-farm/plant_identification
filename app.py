import os
import uuid
import json
from flask import Flask, request, render_template, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- MODIFIED SECTION STARTS HERE ---

# Paths for Keras 3 .keras models
MODEL_PATHS = {
    'chili': 'models/chili_disease_model.keras',
    'tomato': 'models/tomato_disease_model.keras'
}
CLASS_INDICES_PATHS = {
    'chili': 'models/chili_class_indices.json',
    'tomato': 'models/tomato_class_indices.json'
}

# Dictionaries to hold models and class names once they are loaded
models = {}
class_names = {}

def get_model(plant_type):
    """
    Loads a model and its class names if they haven't been loaded yet.
    This is "lazy loading" - it only uses memory when a specific model is needed.
    """
    if plant_type not in models:
        print(f"Loading model for {plant_type}...")
        # Load the model
        model_path = MODEL_PATHS[plant_type]
        models[plant_type] = load_model(model_path, compile=False)
        
        # Load the class names
        class_path = CLASS_INDICES_PATHS[plant_type]
        with open(class_path, 'r') as f:
            class_idx = json.load(f)
        
        # Invert the class index dictionary
        class_names[plant_type] = {int(v): k for k, v in class_idx.items()}
        print(f"Model for {plant_type} loaded successfully.")
    
    return models[plant_type], class_names[plant_type]

def prepare_image(img_path, target_size=(224, 224)):
    """Preprocess uploaded image for prediction."""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def model_predict(filepath, plant_type):
    """Run model prediction and return class + confidence."""
    # Get the model and class names, loading them if necessary
    model, current_class_names = get_model(plant_type)
    
    img = prepare_image(filepath)
    preds = model.predict(img)
    
    class_idx = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds))
    predicted_class = current_class_names.get(class_idx, "Unknown")
    return predicted_class, confidence

# --- MODIFIED SECTION ENDS HERE ---


# Disease solutions (Your existing dictionary goes here)
solution_dict = {
    'chili': {
        "Bacterial-spot": "Use disease-free seeds, copper sprays, and avoid overhead irrigation.",
        "Cercospora-leaf-spot": "Apply fungicides, remove infected leaves, and improve plant spacing.",
        "curl-virus": "Control aphids and whiteflies, remove infected plants.",
        "Healthy-Leaf": "No action needed. Maintain proper care.",
        "Nutrition-deficiency": "Apply balanced fertilizers and improve soil health.",
        "Unlabeled": "No diagnosis available. Upload a clearer image.",
        "White-spot": "Remove affected leaves and apply appropriate fungicides."
    },
    'tomato': {
        "Tomato__Target_Spot": "Remove infected leaves, apply fungicides, and rotate crops to prevent spread.",
        "Tomato__Tomato_mosaic_virus": "Remove and destroy infected plants, disinfect tools, and use virus-resistant varieties.",
        "Tomato__Tomato_YellowLeaf__Curl_Virus": "Control whitefly vectors, remove infected plants, and use resistant varieties.",
        "Tomato_Bacterial_spot": "Use copper-based bactericides and avoid overhead irrigation.",
        "Tomato_Early_blight": "Remove infected leaves, apply fungicides, and rotate crops.",
        "Tomato_healthy": "No action needed. Keep monitoring for signs of disease.",
        "Tomato_Late_blight": "Remove infected plants and apply preventive fungicides.",
        "Tomato_Leaf_Mold": "Increase air circulation, avoid overhead watering, and use fungicides.",
        "Tomato_Septoria_leaf_spot": "Remove infected leaves and use fungicides.",
        "Tomato_Spider_mites_Two_spotted_spider_mite": "Use miticides and encourage beneficial predatory insects."
    }
}


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        plant_type = request.form.get('plant_type')
        if plant_type not in ['chili', 'tomato']:
            return render_template('index.html', prediction="Invalid plant type", plant=None, solution=None, image_url=None)

        file = request.files.get('file')
        if not file or file.filename == '':
            return render_template('index.html', prediction="No file uploaded", plant=None, solution=None, image_url=None)

        # Save uploaded file with unique name
        filename = f"{uuid.uuid4().hex}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            predicted_class, confidence = model_predict(filepath, plant_type)
            if not predicted_class:
                raise ValueError("Prediction class not found in class names")

            solution = solution_dict.get(plant_type, {}).get(predicted_class, "No solution found.")
            prediction_text = f"{predicted_class.replace('_', ' ')} ({confidence*100:.1f}% confidence)" if confidence > 0.1 else f"Uncertain: {predicted_class.replace('_', ' ')}"

        except Exception as e:
            return render_template('index.html', prediction=f"Error: {e}", plant=None, solution=None, image_url=None)

        return render_template('index.html',
                               prediction=prediction_text,
                               plant=plant_type.capitalize(),
                               solution=solution,
                               image_url=url_for('static', filename=f'uploads/{filename}'))

    return render_template('index.html', prediction=None, plant=None, solution=None, image_url=None)

# You don't need the __main__ block for Render with Gunicorn
# But it's good to keep for local testing
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)