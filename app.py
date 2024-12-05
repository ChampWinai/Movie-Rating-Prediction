import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# โหลดโมเดลที่ฝึกไว้
model = load_model('my_model.h5')

# รายชื่อคอลัมน์ genre ที่โมเดลคาดหวัง
all_genres_columns = ['genre_Action', 'genre_Comedy', 'genre_Drama', 'genre_Biography', 'genre_Crime', 
                      'genre_History', 'genre_Romance', 'genre_Science', 'genre_Thriller', 'genre_Mystery', 
                      'genre_Fantasy', 'genre_Adventure', 'genre_Horror', 'genre_Documentary', 'genre_Animation', 
                      'genre_Musical', 'genre_Family', 'genre_Sport', 'genre_War', 'genre_Western', 'genre_News', 
                      'genre_TalkShow', 'genre_Religion', 'genre_Food', 'genre_Music', 'genre_Tech', 'genre_Education', 
                      'genre_Politics', 'genre_Art', 'genre_Lifestyle']

# ฟังก์ชันเตรียมข้อมูล
def prepare_data(duration, votes, genres):
    # แปลง genre เป็น one-hot encoding
    genre_dummies = pd.get_dummies(genres, prefix='genre')
    
    # เติมคอลัมน์ที่ขาดหายไปและลบคอลัมน์ที่ไม่ต้องการ
    genre_dummies = genre_dummies.reindex(columns=all_genres_columns, fill_value=0)
    
    # สร้าง DataFrame สำหรับ duration และ votes
    data = pd.DataFrame({
        'duration': [duration],
        'votes': [votes]
    })
    
    # รวมข้อมูลทั้งหมด
    data = pd.concat([data, genre_dummies], axis=1)
    
    # ตรวจสอบขนาดของข้อมูล
    print(f"Shape of data: {data.shape}")  # ควรเป็น (1, 29)
    
    # หากคอลัมน์มากกว่า 29, ตัดคอลัมน์ที่ไม่จำเป็นออก
    if data.shape[1] > 29:
        data = data.iloc[:, :29]
        print("Excess columns removed, new shape: ", data.shape)
    
    # ปรับสเกลข้อมูล duration และ votes
    scaler = StandardScaler()
    data[['duration', 'votes']] = scaler.fit_transform(data[['duration', 'votes']])
    
    return data

# Streamlit UI
st.title('Movie Rating Prediction')

st.write("""
    **Enter details of a movie to predict its rating:**
""")

# รับข้อมูลจากผู้ใช้
duration = st.number_input('Duration (in minutes)', min_value=1, max_value=300, value=120)
votes = st.number_input('Number of votes', min_value=1, value=5000)
genres = st.selectbox('Select Genres', 
                        ['Action', 'Comedy', 'Drama', 'Biography', 'Crime', 'History', 'Romance',
                         'Science', 'Thriller', 'Mystery', 'Fantasy', 'Adventure', 'Horror', 
                         'Documentary', 'Animation', 'Musical', 'Family', 'Sport', 'War', 
                         'Western', 'News', 'TalkShow', 'Religion', 'Food', 'Music', 'Tech', 
                         'Education', 'Politics', 'Art', 'Lifestyle'])

# เมื่อผู้ใช้กดปุ่มทำนาย
if st.button('Predict Rating'):
    # เตรียมข้อมูลที่ได้จากผู้ใช้
    input_data = prepare_data(duration, votes, genres)
    
    # ทำนายผลลัพธ์
    predicted_rating = model.predict(input_data)
    
    # แสดงผลลัพธ์
    st.write(f"Predicted Rating: {predicted_rating[0][0]:.2f}")
