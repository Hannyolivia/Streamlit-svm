import streamlit as st
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Memuat model SVM dan TF-IDF vectorizer yang telah disimpan
svm_classifier = joblib.load('svm_model_smote.pkl')
tfidf_vectorizer = joblib.load('vectorizer.pkl')

# Fungsi untuk melakukan prediksi terhadap input teks
def predict_sentiment(input_text):
    input_tfidf = tfidf_vectorizer.transform([input_text])  # Mengubah teks input menjadi fitur TF-IDF
    prediction = svm_classifier.predict(input_tfidf)  # Prediksi menggunakan model SVM
    return 'Positif' if prediction[0] == 1 else 'Negatif'

# Header dan penjelasan aplikasi
st.title("ANALISIS SENTIMEN 🎮")

st.markdown(
    """
     **Aplikasi Analisis Sentimen** menggunakan model **SVM** untuk memprediksi sentimen dari teks yang Anda masukkan. 
    
    📌 **Cara Penggunaan:**
    1. Masukkan kalimat pada kolom input.
    2. Klik tombol **"Prediksi Sentimen"**.
    3. Lihat hasil prediksi apakah **Positif** 😊 atau **Negatif** 😠.
    """,
    unsafe_allow_html=True
)

# Input dari pengguna
user_input = st.text_input("📝 Masukkan kalimat untuk diprediksi:", "")

# Tombol untuk memulai prediksi
if st.button('🔍 Prediksi Sentimen'):
    if user_input:  # Pastikan ada input dari pengguna
        sentiment = predict_sentiment(user_input)  # Prediksi sentimen
        
        # Menentukan warna dan emoji berdasarkan sentimen
        if sentiment == "Positif":
            st.success(f"😊 Sentimen dari kalimat: **{sentiment}**")
        else:
            st.error(f"😠 Sentimen dari kalimat: **{sentiment}**")
    else:
        st.warning("⚠️ Silakan masukkan kalimat terlebih dahulu.")


st.write("\n\n---\n**Hanny Olivia 21.12.2005**")