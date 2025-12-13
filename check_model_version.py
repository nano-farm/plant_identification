import zipfile
import json
import os

# 1. Fix the path syntax using 'r' (raw string) to avoid the SyntaxWarning
model_path = r"models\chili_disease_model_oversampled.keras"

print(f"🔎 Checking model file: {model_path}")

if not os.path.exists(model_path):
    print("❌ Error: File not found at that path.")
else:
    try:
        # .keras files are actually ZIP files. We open them with zipfile.
        with zipfile.ZipFile(model_path, "r") as archive:
            
            # Check for metadata.json (standard in Keras v3)
            if "metadata.json" in archive.namelist():
                with archive.open("metadata.json") as f:
                    meta = json.load(f)
                    print("\n✅ Metadata Found:")
                    print(f"   • Keras Version: {meta.get('keras_version', 'Unknown')}")
                    print(f"   • Date Saved:    {meta.get('date_saved', 'Unknown')}")
            
            # Check for config.json (sometimes contains version info too)
            elif "config.json" in archive.namelist():
                with archive.open("config.json") as f:
                    config = json.load(f)
                    print("\n⚠️ No metadata.json, checking config.json...")
                    print(f"   • Keras Version: {config.get('keras_version', 'Unknown')}")
            
            else:
                print("\n❌ Could not find version info inside the .keras archive.")

    except zipfile.BadZipFile:
        print("\n❌ Error: This file is not a valid .keras zip archive.")
        print("   It might be an older .h5 file renamed to .keras.")