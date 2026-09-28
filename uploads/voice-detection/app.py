from flask import Flask, request, render_template
import joblib
import os
import subprocess
from feature_extraction import extract_features
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("voice_model.pkl")


@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        if "audio" in request.files:
            file = request.files["audio"]

            input_path = "temp_input"
            output_path = "temp_output.wav"

            # Save uploaded/recorded file
            file.save(input_path)

            try:
                # Convert any format to WAV using ffmpeg
                subprocess.call(["ffmpeg", "-y", "-i", input_path, output_path])

                # Extract features
                features = extract_features(output_path)
                features = features.reshape(1, -1)

                # Predict
                prediction = model.predict(features)

                result = "AI Voice" if prediction[0] == 1 else "Human Voice"

            except Exception as e:
                result = f"Error processing audio: {str(e)}"

            finally:
                # Clean temporary files
                if os.path.exists(input_path):
                    os.remove(input_path)
                if os.path.exists(output_path):
                    os.remove(output_path)

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)