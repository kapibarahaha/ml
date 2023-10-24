from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
import numpy as np
from imblearn.over_sampling import RandomOverSampler

app = Flask(__name__)

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
X = df.drop(columns=['Quality'])
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

oversampler_y2 = RandomOverSampler(sampling_strategy='auto', random_state=42)
X_train_resampled_y2, y2_train_resampled = oversampler_y2.fit_resample(X_train, y2_train)

le_fish = LabelEncoder()
le_fish.fit(y2_train_resampled)

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

# สร้างฟังก์ชัน Flask สำหรับรับข้อมูลผู้ใช้และทำนาย
@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        user_input = {
            'DO': float(request.form['DO']),
            'Color': request.form['Color'],
            'Smell': request.form['Smell'],
            'TDS (ppm)': float(request.form['TDS']),
            'EC (microsecond/cm)': float(request.form['EC']),
            'PH': float(request.form['PH']),
            'Area': float(request.form['Area']),
            'time': float(request.form['time'])
        }

        user_input['Color'] = label_encoders['Color'].transform([user_input['Color']])[0]
        user_input['Smell'] = label_encoders['Smell'].transform([user_input['Smell']])[0]

        user_input_df = pd.DataFrame([user_input])
        user_input_df['fish'] = 0  # ใส่ค่าเริ่มต้นที่ต้องการให้

        user_input_np = user_input_df.to_numpy()
        user_input_scaled = scaler.transform(user_input_np)

        quality_prediction = ensemble_classifier.predict(user_input_scaled)
        quantity_prediction = ensemble_classifier.predict(user_input_scaled)

        quality_prediction_label = label_encoders['Quality'].inverse_transform(quality_prediction)
        quantity_prediction_label = le_fish.inverse_transform(quantity_prediction)

        return render_template('result.html', quality=quality_prediction_label[0], quantity=quantity_prediction_label[0])

    return render_template('input.html')

if __name__ == '__main__':
    app.run(debug=True,port=8088)