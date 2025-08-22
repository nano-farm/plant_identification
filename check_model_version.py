import h5py

model_path = "models/chili_disease_model.h5"  # change to tomato model if you want

with h5py.File(model_path, "r") as f:
    print("🔎 Checking model file:", model_path)
    if "keras_version" in f.attrs:
        print("✅ Keras version used to save:", f.attrs["keras_version"])
    if "backend" in f.attrs:
        print("✅ Backend:", f.attrs["backend"])
    if "tensorflow_version" in f.attrs:
        print("✅ TensorFlow version:", f.attrs["tensorflow_version"])
