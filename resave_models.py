from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the exact same model file you deployed
model = load_model('models/tomato_disease_model.keras', compile=False)

# This is your prepare_image function from app.py
def prepare_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Crucial normalization step
    return img_array

# Use a specific training image
image_path = r'C:\Users\seena\plant_identification\data\train\tomato\Tomato__Tomato_mosaic_virus\0a7cc59f-b2b0-4201-9c4a-d91eca5c03a3___PSU_CG 2230.JPG'

# Prepare and predict
processed_image = prepare_image(image_path)
predictions = model.predict(processed_image)

print(f"Confidence: {np.max(predictions) * 100:.2f}%")
print(f"Predicted class index: {np.argmax(predictions)}")