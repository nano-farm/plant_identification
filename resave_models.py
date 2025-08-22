from tensorflow.keras.models import load_model

# Resave chili model
chili_model = load_model("models/chili_disease_model.h5", compile=False)
chili_model.save("models/chili_disease_model.h5")  # overwrite

# Resave tomato model
tomato_model = load_model("models/tomato_disease_model.h5", compile=False)
tomato_model.save("models/tomato_disease_model.h5")  # overwrite
