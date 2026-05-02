from flask import Flask, jsonify, render_template, request

from spam_detector.predictor import SpamPredictor

app = Flask(__name__, template_folder="../templates", static_folder="../static")
predictor = SpamPredictor()


@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
    text = request.json.get("text", "") if request.is_json else request.form.get("sms_text", "")
    threshold = float(request.args.get("threshold", 0.50))

    if not text or not text.strip():
        payload = {"error": "SMS text is required."}
        return (jsonify(payload), 400) if request.is_json else (render_template("result.html", error=payload["error"]), 400)

    result = predictor.predict(text, threshold=threshold)
    if request.is_json:
        return jsonify(result)
    return render_template("result.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
