import streamlit as st
import pickle

# -----------------------------
# تحميل الموديل
# -----------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# -----------------------------
# عنوان الصفحة
# -----------------------------
st.set_page_config(page_title="Autism Prediction App", page_icon="🧠", layout="centered")

st.title("🧠 Autism Prediction System")
st.write("ادخل بيانات الطفل عشان نعمل التنبؤ 👇")

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
# تحويل الخيارات لأرقام (نفس الكود اللي استعملتيه وقت التدريب)
# -----------------------------
gender_num = 1 if gender == "Male" else 0
smoke_num = 1 if smoke == "Yes" else 0
family_num = 1 if family_history == "Yes" else 0
premature_num = 1 if premature == "Yes" else 0
complications_num = 1 if complications == "Yes" else 0
speech_num = 1 if speech_delay == "Yes" else 0
eye_num = 1 if eye_contact == "Yes" else 0
social_num = 1 if social_interaction == "Yes" else 0

# مصفوفة الإدخال
data = [[
    gender_num, age_child, mother_age, father_age,
    smoke_num, family_num, premature_num, complications_num,
    speech_num, eye_num, social_num
]]

# -----------------------------
# التنبؤ
# -----------------------------
if st.button("🔮 اعمل التوقع"):
    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("⚠️ النتيجة: هناك احتمالية للتوحد")
    else:
        st.success("✅ النتيجة: لا يوجد توحد")


