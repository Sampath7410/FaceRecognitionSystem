from flask import Flask, render_template, request
import os
from predict import predict_image
import json

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

num_classes = len(class_names)

# Load dataset sizes from `train_model.py`
with open("dataset_info.json", "r") as f:
    dataset_info = json.load(f)

num_train_samples = dataset_info["train_samples"]
num_test_samples = dataset_info["test_samples"]
total_samples = dataset_info["total_samples"]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    uploaded_image = None

    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded", 400
        file = request.files["file"]
        if file.filename == "":
            return "No selected file", 400

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        predicted_class = predict_image(filepath)
        return render_template(
            "index.html",
            uploaded_image=filepath,
            prediction=predicted_class,
            num_classes=num_classes,
            num_train_samples=num_train_samples,
            num_test_samples=num_test_samples,
            total_samples=total_samples,
        )

    return render_template(
        "index.html",
        uploaded_image=None,
        prediction=None,
        num_classes=num_classes,
        num_train_samples=num_train_samples,
        num_test_samples=num_test_samples,
        total_samples=total_samples,
    )

if __name__ == "__main__":
    app.run(debug=True)
