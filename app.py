import streamlit as st

st.title("توقع التوحد عند الأطفال 👶🧩")

st.write("أهلا وسهلا! جربي إدخال القيم التالية:")

gender = st.selectbox("الجنس", ["Male", "Female"])
mother_age = st.selectbox("عمر الأم", ["20-30", "30-40"])
smoke = st.radio("هل تدخن الأم؟", ["Yes", "No"])

if st.button("اعمل التوقع"):
    # هون مكان الكود تبع الموديل اللي دربتيه
    st.success("النتيجة: لا يوجد توحد ✅ (هذا مثال فقط)")

