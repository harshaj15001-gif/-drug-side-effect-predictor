from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# ── 1. Dataset ───────────────────────────────────────────────────────────────
data = {
    "Drug": ["Paracetamol","Ibuprofen","Aspirin","Cetirizine","Amoxicillin",
             "Diclofenac","Metformin","Azithromycin","Ciprofloxacin","Dolo650"],
    "Dosage": [500, 400, 300, 10, 250, 50, 500, 500, 500, 650],
    "Age":    [25,  40,  35,  20,  50,  30,  45,  28,  33,  22],
    "SideEffect": ["Nausea","Dizziness","Stomach Pain","Drowsiness","Rash",
                   "Acidity","Weakness","Vomiting","Headache","Nausea"]
}

df = pd.DataFrame(data)

# ── 2. Encode ─────────────────────────────────────────────────────────────────
le_drug = LabelEncoder()
le_side = LabelEncoder()
df["Drug"]       = le_drug.fit_transform(df["Drug"])
df["SideEffect"] = le_side.fit_transform(df["SideEffect"])

# ── 3. Train ──────────────────────────────────────────────────────────────────
X = df[["Drug", "Dosage", "Age"]]
y = df["SideEffect"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)
print("Model trained!")

drug_list = ["Paracetamol","Ibuprofen","Aspirin","Cetirizine","Amoxicillin",
             "Diclofenac","Metformin","Azithromycin","Ciprofloxacin","Dolo650"]

# ── 4. Routes ─────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html", drugs=drug_list)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        drug_name = request.form["drug"]
        dosage    = int(request.form["dosage"])
        age       = int(request.form["age"])

        drug_encoded = le_drug.transform([drug_name])[0]
        input_data   = pd.DataFrame([[drug_encoded, dosage, age]],
                                    columns=["Drug", "Dosage", "Age"])
        pred_encoded = model.predict(input_data)[0]
        side_effect  = le_side.invers