import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Autism Prediction App")
st.title("🧩 نظام توقع التوحد عند الأطفال")
st.write("ادخلي بيانات الطفل لنقوم بمحاولة التنبؤ:")

# -----------------------------
# إدخالات المستخدم
# -----------------------------
gender = st.selectbox("جنس الطفل:", ["Male", "Female"])
age_child = st.number_input("عمر الطفل:", min_value=0, max_value=18, value=5)
mother_age = st.number_input("عمر الأم:", min_value=15, max_value=60, value=30)
father_age = st.number_input("عمر الأب:", min_value=15, max_value=80, value=35)

smoke = st.selectbox("هل الأم كانت تدخن أثناء الحمل؟", ["No", "Yes"])
family_history = st.selectbox("هل يوجد تاريخ عائلي للتوحد؟", ["No", "Yes"])
premature = st.selectbox("هل الطفل ولد قبل أوانه؟", ["No", "Yes"])
complications = st.selectbox("هل كانت هناك مضاعفات عند الولادة؟", ["No", "Yes"])

speech_delay = st.selectbox("هل يوجد تأخر في الكلام؟", ["No", "Yes"])
eye_contact = st.selectbox("هل الطفل يتجنب التواصل البصري؟", ["No", "Yes"])
social_interaction = st.selectbox("هل الطفل يعاني من صعوبات اجتماعية؟", ["No", "Yes"])

# -----------------------------
# بيانات وهمية لتدريب الموديل داخل الكود
# (لنستخدمها مباشرة بدون pickle)
# -----------------------------
# مثال سريع للبيانات، استخدمي بياناتك الحقيقية لو عندك
data_dict = {
    "Gender": ["Male","Female","Male","Female"],
    "age_child": [5,6,4,7],
    "mother_age": [30,32,28,35],
    "father_age": [35,36,33,40],
    "smoke": ["No","Yes","No","No"],
    "family_history": ["No","Yes","No","No"],
    "premature": ["No","Yes","No","No"],
    "complications": ["No","No","Yes","No"],
    "speech_delay": ["No","Yes","No","No"],
    "eye_contact": ["No","Yes","No","No"],
    "social_interaction": ["No","Yes","No","No"],
    "autistic": [0,1,0,0]
}
df = pd.DataFrame(data_dict)

# -----------------------------
# تجهيز البيانات
# -----------------------------
features = df.drop("autistic", axis=1)
target = df["autistic"]

# تحويل القيم التصنيفية لأرقام
features = features.apply(LabelEncoder().fit_transform)

# تدريب الموديل
model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(features, target)

# -----------------------------
# تحويل مدخلات المستخدم لأرقام
# -----------------------------
input_df = pd.DataFrame([[
    gender, age_child, mother_age, father_age,
    smoke, family_history, premature, complications,
    speech_delay, eye_contact, social_interaction
]], columns=features.columns)

input_encoded = input_df.apply(LabelEncoder().fit_transform)

# -----------------------------
# زر التنبؤ
# -----------------------------
if st.button("اعمل التوقع"):
    prediction = model.predict(input_encoded)
    if prediction[0] == 1:
        st.error("⚠️ النتيجة: هناك احتمالية للتوحد")
    else:
        st.success("✅ النتيجة: لا يوجد توحد")

if st.button("اعمل التوقع"):
    prediction = model.predict(input_encoded)
    proba = model.predict_proba(input_encoded)
    confidence = proba[0][prediction[0]] * 100

    if prediction[0] == 1:
        st.error(f"⚠️ النتيجة: هناك احتمالية للتوحد")
    else:
        st.success(f"✅ النتيجة: لا يوجد توحد")
    
    st.write(f"نسبة الثقة في التنبؤ: {confidence:.2f}%")
