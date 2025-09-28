import streamlit as st

# عنوان التطبيق
st.title("🧩 توقع التوحد عند الأطفال 👶")

st.write("أهلا وسهلا! قومي بإدخال البيانات التالية لنقوم بمحاولة التوقع:")

# --- إدخالات المستخدم ---
gender = st.selectbox("الجنس", ["Male", "Female"])
age_child = st.slider("عمر الطفل (بالسنوات)", 1, 10, 3)
mother_age = st.selectbox("عمر الأم", ["<20", "20-30", "30-40", "40+"])
father_age = st.selectbox("عمر الأب", ["<25", "25-35", "35-45", "45+"])
smoke = st.radio("هل تدخن الأم؟", ["Yes", "No"])
family_history = st.radio("هل يوجد تاريخ عائلي للتوحد؟", ["Yes", "No"])
premature = st.radio("هل كان الطفل مولوداً قبل أوانه (خديج)؟", ["Yes", "No"])
complications = st.radio("هل كان هناك مضاعفات أثناء الحمل/الولادة؟", ["Yes", "No"])
speech_delay = st.radio("هل يعاني الطفل من تأخر في الكلام؟", ["Yes", "No"])
eye_contact = st.radio("هل يحافظ الطفل على التواصل البصري؟", ["Yes", "No"])
social_interaction = st.radio("هل يتفاعل الطفل اجتماعياً مع الآخرين؟", ["Yes", "No"])

# --- زر التوقع ---
if st.button("🔮 اعمل التوقع"):
    # هون رح تحطي الكود تبع الموديل اللي دربتيه
    # مثلا:
    # result = model.predict([gender, age_child, ...])
    
    # لسا مجرد مثال:
    st.success("✅ النتيجة: لا يوجد توحد (مثال فقط)")
    # أو
    # st.error("⚠️ النتيجة: هناك احتمالية للتوحد (مثال فقط)")

