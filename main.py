import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import RandomOverSampler
import streamlit as st

# อ่านข้อมูล
df = pd.read_excel('DatasetEng3.xlsx')

# ใช้ LabelEncoder แปลงคอลัมน์ 'Color', 'Smell', 'Quality', และ 'fish'
label_encoders = {}
categorical_columns = ['Color', 'Smell', 'Quality', 'fish']

for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# แบ่งข้อมูลเป็นชุดฝึกและทดสอบ
X = df.drop(columns=['Quality'])  # เอาคอลัมน์ 'Quality' ออกจาก X
y1 = df['Quality']
y2 = df['fish']

X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    X, y1, y2, test_size=0.2, random_state=42)

# ปรับปรุงตัวแปรเป็นตัวเลข (ถ้าจำเป็น)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

oversampler_y1 = RandomOverSampler(sampling_strategy='auto', random_state=42)
X_train_resampled_y1, y1_train_resampled = oversampler_y1.fit_resample(X_train, y1_train)

# Ensure the LabelEncoder has seen all possible labels in the 'fish' column
le_fish = LabelEncoder()
le_fish.fit(y2_train)  # Fit on the training data


# สร้าง Base Classifiers
base_classifiers = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('xgb', XGBClassifier(n_estimators=100, random_state=42))
]

# สร้าง VotingClassifier โดยรวม Base Classifiers
ensemble_classifier = VotingClassifier(estimators=base_classifiers, voting='hard')

# ฝึกโมเดล VotingClassifier บนชุดข้อมูล y2 ที่ถูก Oversampled
ensemble_classifier.fit(X_train_resampled_y1, y1_train_resampled)

# สร้างส่วนสำหรับกรอกข้อมูล
st.header("ป้อนข้อมูล")
do = st.text_input("DO")
color = st.text_input("Color")
smell = st.text_input("Smell")
tds = st.text_input("TDS (ppm)")
ec = st.text_input("EC (microsecond/cm)")
ph = st.text_input("PH")
area = st.text_input("Area")
time = st.text_input("Time")

if st.button("คำนวณคุณภาพของน้ำและจำนวนปลา"):
    user_input = {
        'DO': float(do),
        'Color': color,
        'Smell': smell,
        'TDS (ppm)': float(tds),
        'EC (microsecond/cm)': float(ec),
        'PH': float(ph),
        'Area': float(area),
        'time': float(time)
    }

    user_input_df = pd.DataFrame([user_input])
    user_input_df['Color'] = label_encoders['Color'].transform(user_input_df['Color'])
    user_input_df['Smell'] = label_encoders['Smell'].transform(user_input_df['Smell'])
    
    # เพิ่มคอลัมน์ 'fish' จาก DataFrame หลัก (X)
    user_input_df['fish'] = X['fish']

    # แปลงเป็น NumPy array
    user_input_np = user_input_df.to_numpy()

    # ปรับขนาดข้อมูลอินพุตของผู้ใช้
    user_input_scaled = scaler.transform(user_input_np)

    # ทำนายคุณภาพของน้ำและจำนวนปลา
    quality_prediction = label_encoders['Quality'].inverse_transform(ensemble_classifier.predict(user_input_scaled))
    fish_prediction = le_fish.inverse_transform(ensemble_classifier.predict(user_input_scaled))

    # แสดงผลลัพธ์
    st.header("ผลลัพธ์")
    st.write(f"คุณภาพของน้ำ: {quality_prediction[0]}")
    st.write(f"จำนวนปลา: {fish_prediction[0]}")
