import streamlit as st
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

@st.cache_data
def load_data():
    autism = pd.read_csv("AUTISMM.csv")  
    autism = autism.drop(['num'], axis=1)

    mylist = ['Male', 'Female']
    list2 = ['30-40', '20-30']

    autism.motherage = autism.motherage.fillna(random.choice(list2))
    autism.smoke = autism.smoke.fillna(value='mode')
    autism.autisticsibiling = autism.autisticsibiling.fillna(value='mode')
    autism.painkillers = autism.painkillers.fillna(value='No')
    autism.Gender = autism.Gender.fillna(random.choice(mylist))
    autism.antibiotics = autism.antibiotics.fillna(value='No')
    autism.radiation = autism.radiation.fillna(value='No')
    autism.twinautistic = autism.twinautistic.fillna(value='no_twin')

    features_num = autism.select_dtypes(exclude=['object'])
    features_cat = autism.select_dtypes(['object'])
    features_cat = features_cat.apply(LabelEncoder().fit_transform)

    autismnew = features_cat.join(features_num)
    return autismnew

autismnew = load_data()
X = autismnew.copy()
y = X.pop('autistic')

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=0)

rf_model = RandomForestClassifier(n_estimators=100, random_state=0)
rf_model.fit(X_train, y_train)

knn_model = neighbors.KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

st.set_page_config(page_title="Autism Prediction", page_icon="🧩")
st.title("🧩 تطبيق توقع احتمالية التوحد")
st.write(" ادخل بيانات الطفل/العائلة لبدء التنبؤ")

autisticsibiling = st.selectbox("هل يوجد أخ/أخت مصاب بالتوحد(اذا كانت الاجابة نعم اضغط 1 واذا كانت لا اضغط 0)؟", [0, 1])
neurologicaldiseases = st.selectbox("(اذا كانت الاجابة نعم اضغط 1 واذا كانت لا اضغط 0)هل يوجد أمراض عصبية بالعائلة؟", [0, 1])
GDM = st.selectbox("سكري الحمل:", [0, 1])
painkillers = st.selectbox("(اذا كانت الاجابة نعم اضغط 1 واذا كانت لا اضغط 0)استخدام مسكنات أثناء الحمل:", [0, 1])
antibiotics = st.selectbox("(اذا كانت الاجابة نعم اضغط 1 واذا كانت لا اضغط 0)استخدام مضادات حيوية:", [0, 1])
smoke = st.selectbox("(اذا كانت الاجابة نعم اضغط 1 واذا كانت لا اضغط 0)هل كانت الام مدخنه ؟", [0, 1])
radiation = st.selectbox("(اذا كانت الاجابة نعم اضغط 1 واذا كانت لا اضغط 0)هل تعرض الجنين الى اشعة ؟", [0, 1])
Gender = st.selectbox("الجنس:", [0, 1])  # 0=Female, 1=Male
twinautistic = st.selectbox("توائم:", [0, 1])
motherage = st.number_input("عمر الأم ((اختار العمر بين 20 و 30 او بين 30 و40)):", min_value=20-30, max_value=30-40)
months = st.number_input("عدد اشهر الحمل:", min_value=1, max_value=10)

input_data = [[
    autisticsibiling, neurologicaldiseases, GDM, painkillers, antibiotics,
    smoke, radiation, Gender, twinautistic, motherage, months
]]

if st.button(" اضغط لبدء التوقع"):
    rf_pred = rf_model.predict(input_data)
    rf_proba = rf_model.predict_proba(input_data)

    knn_pred = knn_model.predict(input_data)
    knn_proba = knn_model.predict_proba(input_data)

    st.subheader("📊 نتائج التوقع:")

    if rf_pred[0] == 1:
        st.error(f"يوجد احتمالية للتوحد ({rf_proba[0][1]*100:.2f}%)")
    else:
        st.success(f"لا يوجد احتمالية توحد ({rf_proba[0][0]*100:.2f}%)")


