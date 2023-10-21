import streamlit as st

st.title("ควยมิว ควยหมวย")
import streamlit as st

st.title('Web Application สำหรับกรอกข้อมูล')

# สร้างส่วนของแบบฟอร์ม
user_input = st.text_input('กรอกข้อมูล:')

if st.button('ส่งข้อมูล'):
    st.write(f'คุณกรอก: {user_input}')
