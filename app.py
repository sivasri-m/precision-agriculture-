from flask import Flask, request, render_template
import pickle
from backend.crop_recommendation import recommend_crop
from backend.crop_yield import predict_yield
from backend.fertilizer_recommendation import recommend_fertilizer

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/crop", methods=["GET", "POST"])  # Allow POST method
def crop():
    if request.method == "POST":
        data = request.form
        crop_name = recommend_crop(int(data["N"]), int(data["P"]), int(data["K"]),
                                   float(data["temperature"]), float(data["humidity"]),
                                   float(data["ph"]), float(data["rainfall"]))
        return render_template("crop.html", prediction=crop_name)
    return render_template("crop.html")

@app.route("/yield", methods=["GET", "POST"])
def yield_prediction():
    if request.method == "POST":
        data = request.form
        predicted_yield = predict_yield(
            int(data["Year"]), 
            float(data["rainfall"]),
            float(data["pesticides"]), 
            float(data["temp"]),
            data["Area"], 
            data["Item"]
        )
        return render_template("yield.html", prediction=predicted_yield)
    return render_template("yield.html")

@app.route("/fertilizer", methods=["GET", "POST"])
def fertilizer():
    if request.method == "POST":
        data = request.form
        fertilizer_name = recommend_fertilizer(
            float(data["Temperature"]),
            float(data["Humidity"]),
            float(data["Moisture"]),
            data["Soil_Type"], 
            data["Crop_Type"],
            int(data["Nitrogen"]),
            int(data["Potassium"]),
            int(data["Phosphorus"])
        )
        return render_template("fertilizer.html", prediction=fertilizer_name)
    return render_template("fertilizer.html")


@app.route("/visualization")
def visualization():
    return render_template("visualization.html")  

@app.route("/analysis")
def analysis():
    return render_template("analysis.html")

@app.route("/map")
def map():
    return render_template("map.html")

@app.route("/view_image/<image>")
def view_image(image):
    image_map = {
        "analysis": "/static/analysis.jpg",
        "map": "/static/map.jpg"
    }
    return render_template("image_view.html", image_src=image_map.get(image, ""))


if __name__ == "__main__":
    app.run(debug=True)
