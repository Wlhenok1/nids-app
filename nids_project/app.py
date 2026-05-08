from flask import Flask, render_template, request, jsonify
import pandas as pd, joblib, json

app = Flask(__name__)
model = joblib.load('models/ids_model.pkl')
features = joblib.load('models/features.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])[features]
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0].max()
    label = "ATTACK" if pred == 1 else "NORMAL"
    return jsonify({"result": label, "confidence": round(float(proba)*100, 2)})

if __name__ == '__main__':
    app.run(debug=True)