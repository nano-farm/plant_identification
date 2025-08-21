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

# ========= Google Drive file IDs =========
DRIVE_FILES = {
    "chili_disease_model.h5": "1Fj6sIVjhjkTRPdnjRDuOF_HMgDsAKWuV",
    "chili_class_indices.json": "1W9jXgq39UXF7BW2RPTk6UJyIkHs3g57q",
    "tomato_disease_model.h5": "1F_9Vof3y9zjlrLCAsAzRk1glsf8pCN5s",
    "tomato_class_indices.json": "16VBgOJ6pJhuG15l6pjGb765VCc_9MmXO"
}

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ========= Function to download from Google Drive =========
def download_from_drive(filename, file_id):
    file_path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(file_path):
        print(f"Downloading {filename} from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", file_path, quiet=False)
    return file_path

# ========= Ensure all models exist locally =========
for fname, fid in DRIVE_FILES.items():
    download_from_drive(fname, fid)

# ========= Load models =========
models = {}
class_names = {}

def load_model_and_classes(name):
    if name == "chili":
        model_path = os.path.join(MODEL_DIR, "chili_disease_model.h5")
        class_path = os.path.join(MODEL_DIR, "chili_class_indices.json")
    elif name == "tomato":
        model_path = os.path.join(MODEL_DIR, "tomato_disease_model.h5")
        class_path = os.path.join(MODEL_DIR, "tomato_class_indices.json")
    else:
        raise ValueError("Unsupported plant type")

    model = load_model(model_path)
    with open(class_path, "r") as f:
        class_idx = json.load(f)
    inv_class_idx = {int(v): k for k, v in class_idx.items()}
    return model, inv_class_idx

models["chili"], class_names["chili"] = load_model_and_classes("chili")
models["tomato"], class_names["tomato"] = load_model_and_classes("tomato")

# ========= Image preprocessing =========
def prepare_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def model_predict(filepath, plant_type):
    img = prepare_image(filepath)
    preds = models[plant_type].predict(img)
    class_idx = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds)
    predicted_class = class_names[plant_type].get(class_idx, None)
    return predicted_class, confidence

# ========= Disease solutions =========
solution_dict = {
    "chili": {
        "Bacterial-spot": "Use disease-free seeds, copper sprays, and avoid overhead irrigation.",
        "Cercospora-leaf-spot": "Apply fungicides, remove infected leaves, and improve plant spacing.",
        "curl-virus": "Control aphids and whiteflies, remove infected plants.",
        "Healthy-Leaf": "No action needed. Maintain proper care.",
        "Nutrition-deficiency": "Apply balanced fertilizers and improve soil health.",
        "Unlabeled": "No diagnosis available. Upload a clearer image.",
        "White-spot": "Remove affected leaves and apply appropriate fungicides."
    },
    "tomato": {
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

# ========= Routes =========
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        plant_type = request.form.get("plant_type")
        if plant_type not in ["chili", "tomato"]:
            return render_template("index.html", prediction="Invalid plant type", plant=None, solution=None, image_url=None)

        if "file" not in request.files:
            return render_template("index.html", prediction="No file uploaded", plant=None, solution=None, image_url=None)

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", prediction="No file selected", plant=None, solution=None, image_url=None)

        # Save file
        filename = f"{uuid.uuid4().hex}_{file.filename}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        try:
            predicted_class, confidence = model_predict(filepath, plant_type)
            solution = solution_dict.get(plant_type, {}).get(predicted_class, "No solution found.")
        except Exception as e:
            return render_template("index.html", prediction=f"Error: {str(e)}", plant=None, solution=None, image_url=None)

        prediction_text = f"{predicted_class.replace('_', ' ')} ({confidence*100:.1f}% confidence)" if confidence > 0.1 else f"Uncertain: {predicted_class.replace('_', ' ')}"

        return render_template("index.html",
                               prediction=prediction_text,
                               plant=plant_type.capitalize(),
                               solution=solution,
                               image_url=url_for("static", filename=f"uploads/{filename}"))

    return render_template("index.html", prediction=None, plant=None, solution=None, image_url=None)

if __name__ == "__main__":
    app.run(debug=True)
