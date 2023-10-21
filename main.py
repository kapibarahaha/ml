import streamlit as st

st.title("Machine Learning Project")
import streamlit as st

st.title('กรอกข้อมูล ตามหัวข้อ')

# สร้างส่วนของแบบฟอร์ม
user_input = st.text_input('กรอกค่า Do:')
user_input = st.text_input('กรอกค่า Color:')
user_input = st.text_input('กรอกค่า Smell:')
user_input = st.text_input('กรอกค่า TDS(ppm):')
user_input = st.text_input('กรอกค่า EC(microsecond/cm):')
user_input = st.text_input('กรอกค่า PH:')
user_input = st.text_input('กรอกค่า Area:')
user_input = st.text_input('กรอกค่า Time:')
if st.button('ส่งข้อมูล'):
    st.write(f'คุณกรอก: {user_input}')
