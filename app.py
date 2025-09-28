import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn.preprocessing import LabelEncoder
import gradio as gr

# --- تحميل البيانات ---
autism = pd.read_csv("AUTISMM.csv")
autism = autism.drop(["num"], axis=1)

# --- معالجة القيم المفقودة ---
mylist = ["Male", "Female"]
list2 = ["30-40", "20-30"]

autism.motherage = autism.motherage.fillna(random.choice(list2))
autism.smoke = autism.smoke.fillna(value="mode")
autism.autisticsibiling = autism.autisticsibiling.fillna(value="mode")
autism.painkillers = autism.painkillers.fillna(value="No")
autism.Gender = autism.Gender.fillna(random.choice(mylist))
autism.antibiotics = autism.antibiotics.fillna(value="No")
autism.radiation = autism.radiation.fillna(value="No")
autism.twinautistic = autism.twinautistic.fillna(value="no_twin")

# --- Encoding ---
features_num = autism.select_dtypes(exclude=["object"])
features_cat = autism.select_dtypes(["object"])
features_cat = features_cat.apply(LabelEncoder().fit_transform)
autismnew = features_cat.join(features_num)

X = autismnew.copy()
y = X.pop("autistic")

# --- تدريب النموذج (RandomForest) ---
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, train_size=0.7, test_size=0.3, random_state=0
)

model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(X_train, y_train)

# --- دالة للتنبؤ ---
def predict_autism(autisticsibiling, neurologicaldiseases, GDM, painkillers,
                   antibioticsr, smoke, radiation, Gender, twinautistic,
                   motherage, months):

    # إدخال البيانات كقائمة بنفس الترتيب
    data = [[autisticsibiling, neurologicaldiseases, GDM, painkillers,
             antibioticsr, smoke, radiation, Gender, twinautistic,
             motherage, months]]

    prediction = model.predict(data)[0]
    proba = model.predict_proba(data)[0]

    return {
        "Prediction": "Autistic" if prediction == 1 else "Not Autistic",
        "Confidence": f"{max(proba)*100:.2f}%"
    }

# --- واجهة Gradio ---
inputs = [
    gr.Number(label="Autistic Sibling"),
    gr.Number(label="Neurological Diseases"),
    gr.Number(label="GDM"),
    gr.Number(label="Painkillers"),
    gr.Number(label="Antibiotics"),
    gr.Number(label="Smoke"),
    gr.Number(label="Radiation"),
    gr.Number(label="Gender"),
    gr.Number(label="Twin Autistic"),
    gr.Number(label="Mother Age"),
    gr.Number(label="Months")
]

outputs = gr.JSON(label="Result")

demo = gr.Interface(fn=predict_autism, inputs=inputs, outputs=outputs,
                    title="Autism Prediction App")

if __name__ == "__main__":
    demo.launch()
