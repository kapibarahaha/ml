import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

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

# Oversampling ข้อมูล y2
oversampler_y2 = RandomOverSampler(sampling_strategy='auto', random_state=42)
X_train_resampled_y2, y2_train_resampled = oversampler_y2.fit_resample(X_train, y2_train)

# Ensure the LabelEncoder has seen all possible labels in the 'fish' column
le_fish = LabelEncoder()
le_fish.fit(y2_train_resampled)  # Fit on the oversampled training data

# สร้าง Base Classifiers
base_classifiers = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('xgb', XGBClassifier(n_estimators=100, random_state=42))
]

# สร้าง VotingClassifier โดยรวม Base Classifiers
ensemble_classifier = VotingClassifier(estimators=base_classifiers, voting='hard')

# ฝึกโมเดล VotingClassifier บนชุดข้อมูล y1 ที่ถูก Oversampled
ensemble_classifier.fit(X_train_resampled_y1, y1_train_resampled)

# ใช้ ensemble_classifier ทำนายผล y1 บนชุดทดสอบ
y1_pred = ensemble_classifier.predict(X_test)

# สร้าง VotingClassifier ใหม่โดยรวม Base Classifiers
ensemble_classifier = VotingClassifier(estimators=base_classifiers, voting='hard')

# ฝึกโมเดล VotingClassifier บนชุดข้อมูล y2 ที่ถูก Oversampled
ensemble_classifier.fit(X_train_resampled_y2, y2_train_resampled)

# ใช้ ensemble_classifier ทำนายผล y2 บนชุดทดสอบ
y2_pred = ensemble_classifier.predict(X_test)

# สำหรับ y1_pred (คุณภาพของน้ำ)
y1_pred_labels = label_encoders['Quality'].inverse_transform(y1_pred)

# สำหรับ y2_pred (จำนวนปลา)
y2_pred_labels = label_encoders['fish'].inverse_transform(y2_pred)

# Streamlit app
st.title("Machine Learning Project")

st.title("กรอกข้อมูลตามหัวข้อ")
user_input_do = st.text_input("กรอกค่า Do:")
user_input_color = st.text_input("กรอกค่า Color:")
user_input_smell = st.text_input("กรอกค่า Smell:")
user_input_tds = st.text_input("กรอกค่า TDS(ppm):")
user_input_ec = st.text_input("กรอกค่า EC(microsecond/cm):")
user_input_ph = st.text_input("กรอกค่า PH:")
user_input_area = st.text_input("กรอกค่า Area:")
user_input_time = st.text_input("กรอกค่า Time:")

if st.button("ส่งข้อมูล"):
    user_input_data = {
        "DO": float(user_input_do),
        "Color": user_input_color,
        "Smell": user_input_smell,
        "TDS (ppm)": float(user_input_tds),
        "EC (microsecond/cm)": float(user_input_ec),
        "PH": float(user_input_ph),
        "Area": float(user_input_area),
        "Time": float(user_input_time),
    }
    
    user_input_df = pd.DataFrame([user_input_data])
    user_input_df['Color'] = label_encoders['Color'].transform(user_input_df['Color'])
    user_input_df['Smell'] = label_encoders['Smell'].transform(user_input_df['Smell'])
    
    # เพิ่มคอลัมน์ 'fish' จาก DataFrame หลัก (X)
    user_input_df['fish'] = X['fish']
    
    # แปลงเป็น NumPy array
    user_input_np = user_input_df.to_numpy()
    
    # ปรับขนาดข้อมูลอินพุตของผู้ใช้
    user_input_scaled = scaler.transform(user_input_np)
    
    # Make predictions for 'คุณภาพของน้ำ'
    quality_prediction = ensemble_classifier.predict(user_input_scaled)
    
    # Make predictions for 'จำนวนปลา'
    quantity_prediction = ensemble_classifier.predict(user_input_scaled)
    
    # สำหรับ y1_pred_labels (คุณภาพของน้ำ)
    y1_pred_labels = label_encoders['Quality'].inverse_transform(quality_prediction)
    
    # สำหรับ y2_pred_labels (จำนวนปลา)
    y2_pred_labels = label_encoders['fish'].inverse_transform(quantity_prediction)
    
    st.write("ผลการทำนายคุณภาพของน้ำ:", y1_pred_labels[0])
    st.write("ผลการทำนายจำนวนปลา:", y2_pred_labels[0])
