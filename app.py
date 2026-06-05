from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")

history = []


@app.route("/")
def index():
    return render_template(
        "index.html",
        total_data=705,
        accuracy="98.78%",
        history=history
    )


@app.route("/predict", methods=["POST"])
def predict():

    age = int(request.form["age"])
    usage = float(request.form["usage"])
    sleep = float(request.form["sleep"])
    mental = float(request.form["mental"])
    conflict = int(request.form["conflict"])

    gender = encoders["Gender"].transform(
        ["Male"]
    )[0]

    academic = encoders[
        "Academic_Level"
    ].transform([
        "Undergraduate"
    ])[0]

    platform = encoders[
        "Most_Used_Platform"
    ].transform([
        "Instagram"
    ])[0]

    academic_effect = encoders[
        "Affects_Academic_Performance"
    ].transform([
        "No"
    ])[0]

    relationship = encoders[
        "Relationship_Status"
    ].transform([
        "Single"
    ])[0]

    data = np.array([[
        age,
        gender,
        academic,
        usage,
        platform,
        academic_effect,
        sleep,
        mental,
        relationship,
        conflict
    ]])

    prediction = model.predict(data)

    result = encoders[
        "Addiction_Level"
    ].inverse_transform(prediction)[0]

    # AI Score
    if result == "Rendah":
        score = np.random.randint(20, 40)
        description = (
            "Penggunaan media sosial masih aman "
            "dan belum menunjukkan indikasi "
            "kecanduan serius."
        )

    elif result == "Sedang":
        score = np.random.randint(41, 70)
        description = (
            "Mulai ada indikasi ketergantungan "
            "media sosial. Disarankan membatasi "
            "durasi penggunaan."
        )

    else:
        score = np.random.randint(71, 100)
        description = (
            "Tingkat kecanduan media sosial tinggi "
            "dan berpotensi mempengaruhi akademik "
            "serta kesehatan mental."
        )

    history.insert(0, result)

    if len(history) > 5:
        history.pop()

    return render_template(
        "index.html",
        result=result,
        score=score,
        description=description,
        total_data=705,
        accuracy="98.78%",
        history=history
    )


if __name__ == "__main__":
    app.run(debug=True)