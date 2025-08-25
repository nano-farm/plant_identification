from tensorflow.keras.models import load_model

# Load old .h5 model
chili_model = load_model("models/chili_disease_model.h5", compile=False)
tomato_model = load_model("models/tomato_disease_model.h5", compile=False)

# Save in Keras V3 native format
chili_model.save("models/chili_disease_model.keras")
tomato_model.save("models/tomato_disease_model.keras")
