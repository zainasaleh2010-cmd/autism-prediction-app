import streamlit as st
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# ----------------------------
# تحميل البيانات
# ----------------------------
@st.cache_data
def load_data():
    autism = pd.read_csv("AUTISMM.csv")  # لازم يكون ملف AUTISMM.csv بنفس مجلد المشروع
    autism = autism.drop(['num'], axis=1)

    # تعويض القيم الفارغة بنفس طريقتك
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

    # فصل رقمي/تصنيفي + ترميز
    features_num = autism.select_dtypes(exclude=['object'])
    features_cat = autism.select_dtypes(['object'])
    features_cat = features_cat.apply(LabelEncoder().fit_transform)

    autismnew = features_cat.join(features_num)
    return autismnew

autismnew = load_data()
X = autismnew.copy()
y = X.pop('autistic')

# ----------------------------
# تدريب الموديل
# ----------------------------
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=0)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=0)
rf_model.fit(X_train, y_train)

# KNN
knn_model = neighbors.KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

# ----------------------------
# واجهة Streamlit
# ----------------------------
st.set_page_config(page_title="Autism Prediction", page_icon="🧩")
st.title("🧩 تطبيق توقع التوحد")
st.write("ادخل بيانات الطفل/العائلة لتجربة التنبؤ:")

# إدخالات المستخدم (لازم نفس ترتيب الأعمدة في X)
autisticsibiling = st.selectbox("هل يوجد أخ/أخت مصاب بالتوحد؟", [0, 1])
neurologicaldiseases = st.selectbox("هل يوجد أمراض عصبية بالعائلة؟", [0, 1])
GDM = st.selectbox("سكري الحمل:", [0, 1])
painkillers = st.selectbox("استخدام مسكنات أثناء الحمل:", [0, 1])
antibiotics = st.selectbox("استخدام مضادات حيوية:", [0, 1])
smoke = st.selectbox("تدخين الأم:", [0, 1])
radiation = st.selectbox("تعرض لإشعاع:", [0, 1])
Gender = st.selectbox("الجنس:", [0, 1])  # 0=Female, 1=Male
twinautistic = st.selectbox("توائم:", [0, 1])
motherage = st.number_input("عمر الأم (كفئة رقمية):", min_value=0, max_value=100, value=30)
months = st.number_input("عمر الطفل بالأشهر:", min_value=0, max_value=240, value=24)

input_data = [[
    autisticsibiling, neurologicaldiseases, GDM, painkillers, antibiotics,
    smoke, radiation, Gender, twinautistic, motherage, months
]]

# ----------------------------
# التنبؤ
# ----------------------------
if st.button("اعمل التوقع"):
    rf_pred = rf_model.predict(input_data)
    rf_proba = rf_model.predict_proba(input_data)

    knn_pred = knn_model.predict(input_data)
    knn_proba = knn_model.predict_proba(input_data)

    st.subheader("📊 نتائج التوقع:")

    if rf_pred[0] == 1:
        st.error(f"RandomForest: يوجد احتمالية للتوحد ({rf_proba[0][1]*100:.2f}%)")
    else:
        st.success(f"RandomForest: لا يوجد توحد ({rf_proba[0][0]*100:.2f}%)")

    if knn_pred[0] == 1:
        st.error(f"KNN: يوجد احتمالية للتوحد ({knn_proba[0][1]*100:.2f}%)")
    else:
        st.success(f"KNN: لا يوجد توحد ({knn_proba[0][0]*100:.2f}%)")

