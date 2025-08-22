import os
import uuid
import json
import gdown
from flask import Flask, request, render_template, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Google Drive file IDs for models and class indices
DRIVE_FILES = {
    "chili_disease_model.h5": "1Fj6sIVjhjkTRPdnjRDuOF_HMgDsAKWuV",
    "chili_class_indices.json": "1W9jXgq39UXF7BW2RPTk6UJyIkHs3g57q",
    "tomato_disease_model.h5": "1F_9Vof3y9zjlrLCAsAzRk1glsf8pCN5s",
    "tomato_class_indices.json": "16VBgOJ6pJhuG15l6pjGb765VCc_9MmXO"
}

# Ensure local models/ directory exists
os.makedirs("models", exist_ok=True)

# Function to download files safely
def download_model_if_missing(filename, file_id):
    local_path = os.path.join("models", filename)
    if not os.path.exists(local_path):
        print(f"Downloading {filename} from Google Drive...")
        url = f"https://drive.google.com/uc?id={file_id}"
        try:
            gdown.download(url, output=local_path, quiet=False)
        except Exception as e:
            print(f"Failed to download {filename}: {e}")
    return local_path

# Download all required files
for filename, file_id in DRIVE_FILES.items():
    download_model_if_missing(filename, file_id)

# Paths for chili
CHILI_MODEL_PATH = 'models/chili_disease_model.h5'
CHILI_CLASS_INDICES_PATH = 'models/chili_class_indices.json'

# Paths for tomato
TOMATO_MODEL_PATH = 'models/tomato_disease_model.h5'
TOMATO_CLASS_INDICES_PATH = 'models/tomato_class_indices.json'

# Load models and class indices at app startup
models = {}
class_names = {}

def load_model_and_classes(name):
    """Load model and corresponding class indices."""
    if name == 'chili':
        model_path, class_path = CHILI_MODEL_PATH, CHILI_CLASS_INDICES_PATH
    elif name == 'tomato':
        model_path, class_path = TOMATO_MODEL_PATH, TOMATO_CLASS_INDICES_PATH
    else:
        raise ValueError("Unsupported plant type")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(class_path):
        raise FileNotFoundError(f"Class indices file not found: {class_path}")

    model = load_model(model_path)
    with open(class_path, 'r') as f:
        class_idx = json.load(f)

    # Convert to {int: class_name}
    inv_class_idx = {int(v): k for k, v in class_idx.items()}
    return model, inv_class_idx

# Initialize models
models['chili'], class_names['chili'] = load_model_and_classes('chili')
models['tomato'], class_names['tomato'] = load_model_and_classes('tomato')

def prepare_image(img_path, target_size=(224, 224)):
    """Load and preprocess image for prediction."""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def model_predict(filepath, plant_type):
    """Run model prediction and return class + confidence."""
    img = prepare_image(filepath)
    preds = models[plant_type].predict(img)
    class_idx = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds))
    predicted_class = class_names[plant_type].get(class_idx, None)
    return predicted_class, confidence

# Disease solution dictionary
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

    # GET request
    return render_template('index.html', prediction=None, plant=None, solution=None, image_url=None)

if __name__ == '__main__':
    app.run(debug=True)
