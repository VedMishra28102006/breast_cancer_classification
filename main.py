import os
from flask import Flask, request, render_template
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
  context = {}
  if request.method == "POST":
    mean_radius = float(request.form["mean_radius"])
    mean_texture = float(request.form["mean_texture"])
    mean_perimeter = float(request.form["mean_perimeter"])
    mean_area = float(request.form["mean_area"])
    mean_smoothness = float(request.form["mean_smoothness"])
    mean_compactness = float(request.form["mean_compactness"])
    mean_concavity = float(request.form["mean_concavity"])
    mean_concave_points = float(request.form["mean_concave_points"])
    mean_symmetry = float(request.form["mean_symmetry"])
    mean_fractal_dimension = float(request.form["mean_fractal_dimension"])

    breast_cancer_data = load_breast_cancer()
    df = pd.DataFrame(breast_cancer_data.data, columns=breast_cancer_data.feature_names)
    df["target"] = breast_cancer_data.target 
    X = df[["mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness", "mean compactness", "mean concave points", "mean symmetry", "mean fractal dimension"]]
    y = df["target"]
    knn = KNeighborsClassifier()
    knn.fit(X, y)
    input_data = pd.DataFrame([[mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness, mean_compactness, mean_concave_points, mean_symmetry, mean_fractal_dimension]], columns=["mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness", "mean compactness", "mean concave points", "mean symmetry", "mean fractal dimension"])
    predicted_class = knn.predict(input_data)
    classes = ["malignant", "benign"]
    context["prediction"] = classes[predicted_class[0]]

  return render_template("index.html", context=context)

if __name__ == "__main__":
  host = os.getenv("IP", "0.0.0.0")
  port = int(os.getenv("PORT", 5000))
  app.run(host=host, port=port)