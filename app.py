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

# Paths for Keras models
MODEL_PATHS = {
    'chili': 'models/chili_disease_model_oversampled.keras',
    'tomato': 'models/tomato_disease_model_oversampled.keras'
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
    """
    if plant_type not in models:
        print(f"Loading model for {plant_type}...")
        model_path = MODEL_PATHS[plant_type]
        models[plant_type] = load_model(model_path, compile=False)
        
        class_path = CLASS_INDICES_PATHS[plant_type]
        with open(class_path, 'r') as f:
            class_idx = json.load(f)
        
        class_names[plant_type] = {int(v): k for k, v in class_idx.items()}
        print(f"Model for {plant_type} loaded successfully.")
    
    return models[plant_type], class_names[plant_type]

# FIXED: Updated target_size to match the 224x224 training dimensions
def prepare_image(img_path, target_size=(224, 224)):
    """Preprocess uploaded image for prediction."""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def model_predict(filepath, plant_type):
    """Run model prediction and return class + confidence."""
    model, current_class_names = get_model(plant_type)
    
    img = prepare_image(filepath)
    preds = model.predict(img)
    
    class_idx = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds))
    predicted_class = current_class_names.get(class_idx, "Unknown")
    return predicted_class, confidence

# Disease solutions
solution_dict = {
"chili": {
    "Bacterial-spot": "Apply copper-based bactericides (like Copper Oxychloride) or a Copper-Mancozeb mixture. Avoid overhead irrigation to keep foliage dry, as bacteria spread in water drops. Remove infected debris and rotate crops to non-solanaceous plants.",
    "Cercospora-leaf-spot": "Spray with Chlorothalonil or Mancozeb at the first sign of spots. Improve air circulation by spacing plants properly. Remove and destroy fallen leaves to reduce fungal spores.",
    "curl-virus": "There is no cure for the virus itself; remove and burn infected plants immediately to prevent spread. Control whitefly vectors using yellow sticky traps, neem oil, or Imidacloprid sprays.",
    "Healthy-Leaf": "Maintain a regular watering and fertilization schedule. Inspect plants weekly for early signs of pests or disease.",
    "Nutrition-deficiency": "Apply a balanced NPK fertilizer. For specific symptoms (e.g., yellowing), use foliar sprays containing micronutrients like Magnesium, Calcium, or Iron. Check soil pH to ensure nutrient uptake.",
    "Unlabeled": "Diagnosis unclear. Please upload a clearer image with the leaf flat and well-lit.",
    "White-spot": "Likely fungal (e.g., Alternaria or Cercospora). Prune heavily affected leaves. Apply broad-spectrum fungicides like Mancozeb or Copper hydroxide. Avoid water accumulation on leaves."
  },
        "tomato": {
            "Tomato___Target_Spot": "Apply fungicides like chlorothalonil or mancozeb at the first sign of symptoms. Improve air circulation by pruning lower leaves and staking plants to keep foliage off the ground. Avoid overhead irrigation to keep leaves dry.",
            "Tomato___Tomato_mosaic_virus": "There is no cure; remove and destroy infected plants immediately to prevent spread. disinfect tools with a bleach solution (1:9 ratio) between plants. Plant resistant varieties labeled 'ToMV' or 'TMV' and control weeds that may harbor the virus.",
            "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Control whitefly populations using yellow sticky traps, insecticidal soaps, or neem oil, as they transmit the virus. Remove and destroy infected plants immediately. Use reflective mulches to repel whiteflies and plant resistant varieties.",
            "Tomato___Bacterial_spot": "Apply copper-based bactericides mixed with mancozeb for better control. Avoid overhead watering and working in the garden when plants are wet. Rotate crops to non-solanaceous plants (avoid peppers and eggplants) for at least 2-3 years.",
            "Tomato___Early_blight": "Apply bio-fungicides like Bacillus subtilis or chemical fungicides containing chlorothalonil or copper. Mulch around the base of plants to prevent soil spores from splashing onto leaves. Prune infected lower leaves and rotate crops every 2-3 years.",
            "Tomato___healthy": "Maintain a regular watering schedule and inspect plants weekly for early signs of pests or disease. Ensure proper spacing for airflow and apply compost to boost plant immunity.",
            "Tomato___Late_blight": "This is a fast-spreading, destructive disease; remove and bag infected plants immediately to stop spores from spreading. Apply preventative fungicides like chlorothalonil or copper if cool, wet weather is forecast. Avoid overhead irrigation.",
            "Tomato___Leaf_Mold": "Maximize air circulation by spacing plants widely and pruning excessive foliage. Water only at the base of the plant to keep leaves dry. If severe, apply fungicides approved for leaf mold control.",
            "Tomato___Septoria_leaf_spot": "Remove fallen leaves and debris from the garden area to reduce overwintering spores. Apply fungicides containing chlorothalonil or copper. Mulch the soil to stop spores from splashing up onto the lower leaves.",
            "Tomato___Spider_mites Two-spotted_spider_mite": "Spray plants with a strong stream of water to dislodge mites. Apply insecticidal soap, neem oil, or horticultural oil to undersides of leaves. Introduce beneficial insects like ladybugs or predatory mites."
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

        filename = f"{uuid.uuid4().hex}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            predicted_class, confidence = model_predict(filepath, plant_type)
            
            CONFIDENCE_THRESHOLD = 0.70

            if confidence >= CONFIDENCE_THRESHOLD:
                prediction_text = f"{predicted_class.replace('_', ' ')} ({confidence*100:.1f}% confidence)"
                solution = solution_dict.get(plant_type, {}).get(predicted_class, "No solution found.")
            else:
                prediction_text = f"Uncertain Diagnosis ({confidence*100:.1f}% confidence)"
                solution = "The model is not confident enough to make a diagnosis. Please try a clearer image taken in good, natural light."

        except Exception as e:
            print(f"An error occurred: {e}")
            prediction_text = "An error occurred during prediction."
            solution = "Please try again with a different image."


        return render_template('index.html',
                               prediction=prediction_text,
                               plant=plant_type.capitalize(),
                               solution=solution,
                               image_url=url_for('static', filename=f'uploads/{filename}'))

    return render_template('index.html', prediction=None, plant=None, solution=None, image_url=None)

# For local testing
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

