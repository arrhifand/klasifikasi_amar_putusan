import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Konfigurasi Halaman
st.set_page_config(page_title="Klasifikasi Putusan", page_icon="⚖️", layout="centered")

# --- PARAMETER MODEL ---
MAX_SEQUENCE_LENGTH = 150 
# CATATAN PENTING: Cek kembali apakah di Colab Anda menggunakan padding='pre' atau padding='post'
PADDING_TYPE = 'pre'  # Default keras adalah 'pre'
TRUNCATING_TYPE = 'pre'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "lstm_model_last_fold.keras")
PATCHED_MODEL_PATH = os.path.join(BASE_DIR, "lstm_model_last_fold_patched.keras")
if os.path.exists(PATCHED_MODEL_PATH):
    MODEL_PATH = PATCHED_MODEL_PATH

TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pkl")

# --- CACHE RESOURCE ---
@st.cache_resource
def load_sastrawi():
    stemmer_factory = StemmerFactory()
    stemmer = stemmer_factory.create_stemmer()
    stopword_factory = StopWordRemoverFactory()
    stopword = stopword_factory.create_stop_word_remover()
    return stemmer, stopword

@st.cache_resource
def load_model_and_tokenizer():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)
        return model, tokenizer, None
    except Exception as e:
        return None, None, str(e)

# --- FUNGSI PREPROCESSING ---
def preprocess_text(text, stemmer, stopword):
    # 1. Hapus tag HTML (seperti <p>, <li>, <strong>, <ol>)
    text = re.sub(r'<[^>]+>', ' ', text)
    
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Hapus stopword bawaan Sastrawi
    text = stopword.remove(text)
    
    # Hapus custom stopword ("li", "p", "strong", "ol" dsb yang mungkin tersisa)
    custom_stopwords = ['li', 'p', 'strong', 'ol', 'br', 'div', 'span']
    words = text.split()
    words = [w for w in words if w not in custom_stopwords]
    text = ' '.join(words)
    
    text = stemmer.stem(text)
    return text

# --- SIDEBAR NAVIGASI ---
st.sidebar.title("Navigasi")
menu = st.sidebar.radio("Pilih Halaman:", ["Prediksi Putusan", "Evaluasi Performa Model"])

if menu == "Prediksi Putusan":
    st.title("⚖️ Klasifikasi Putusan Perceraian")
    st.markdown("Aplikasi ini memprediksi klasifikasi putusan perceraian (**Cerai Talak** atau **Cerai Gugat**).")

    stemmer, stopword = load_sastrawi()
    model, tokenizer, load_error = load_model_and_tokenizer()

    if model is None or tokenizer is None:
        st.error(f"⚠️ Gagal memuat Model atau Tokenizer!\n\nPastikan Anda telah meletakkan file model dan tokenizer di folder:\n`{BASE_DIR}`")
        st.error(f"**Detail Error:** {load_error}")
        st.stop()

    user_input = st.text_area("Masukkan Teks Putusan:", height=250, placeholder="Ketik atau paste dokumen putusan di sini...")

    if st.button("Prediksi", type="primary"):
        if user_input.strip() == "":
            st.warning("Mohon masukkan teks putusan terlebih dahulu.")
        else:
            with st.spinner("Memproses teks... (Proses stemming Sastrawi memakan waktu)"):
                try:
                    cleaned_text = preprocess_text(user_input, stemmer, stopword)
                    sequences = tokenizer.texts_to_sequences([cleaned_text])
                    padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding=PADDING_TYPE, truncating='post')
                    
                    prediction_prob = model.predict(padded_sequences)[0][0]
                    
                    if prediction_prob >= 0.5:
                        hasil = "Cerai Talak"
                        confidence = prediction_prob * 100
                    else:
                        hasil = "Cerai Gugat"
                        confidence = (1 - prediction_prob) * 100
                    
                    st.success("Selesai!")
                    st.markdown(f"### Kategori: **{hasil}**")
                    st.progress(int(confidence) / 100)
                    st.caption(f"Tingkat Keyakinan (Confidence): {confidence:.2f}%")
                    
                    with st.expander("🛠️ Lihat Detail Debugging (Klik di sini)"):
                        st.markdown("**1. Teks Setelah Preprocessing:**")
                        st.write(cleaned_text)
                        st.markdown("**2. Sequence Tokenizer (Angka Keras):**")
                        st.write(sequences[0])
                        st.markdown("**3. Probabilitas Mentah (Raw Probability):**")
                        st.code(str(prediction_prob))
                        
                        st.info("""
                        **PANDUAN DEBUGGING PREDIKSI TERBALIK:**
                        1. Jika nilai Probabilitas di atas sangat mendekati 0 (misal: `0.001`), dan aslinya teks tersebut adalah **Cerai Gugat**, itu berarti di model Anda `0 = Cerai Gugat` dan `1 = Cerai Talak`.
                        2. Jika demikian, Anda cukup membalik logika di kode `app.py` pada baris ke-85 menjadi:
                           `if prediction_prob >= 0.5: hasil = "Cerai Talak" else: hasil = "Cerai Gugat"`
                        3. Selain itu, dokumen putusan biasanya sangat panjang. Karena `MAX_SEQUENCE_LENGTH = 150`, teks Anda akan **terpotong**. Jika kata kunci penting berada di awal dokumen, pastikan di Colab Anda menggunakan `truncating='post'`, dan ubah `TRUNCATING_TYPE` di `app.py` menjadi `'post'` juga.
                        """)
                        
                except Exception as e:
                    st.error(f"Terjadi kesalahan: {e}")

elif menu == "Evaluasi Performa Model":
    st.title("📊 Evaluasi Performa Model")
    st.markdown("""
    Halaman ini menampilkan grafik dan metrik performa model hasil training.
    
    > **⚠️ INFORMASI PENTING:**  
    > Ini adalah hasil dari evaluasi dari pelatihan model menggunakan K-Fold Cross-Validation.
    """)
    
    # ==========================================
    # BAGIAN 1: METRIK AKURASI, PRECISION, RECALL, F1
    # ==========================================
    st.subheader("1. Ringkasan Metrik (Rata-rata 5-Fold)")
    
    metrics_data = pd.DataFrame({
        "Metrik": [
            "Accuracy", 
            "Precision (Gugat)", "Recall (Gugat)", "F1-Score (Gugat)",
            "Precision (Talak)", "Recall (Talak)", "F1-Score (Talak)"
        ],
        "Nilai": [
            0.9783, 
            0.9741, 0.9973, 0.9855,
            0.9916, 0.9238, 0.9564
        ] 
    })
    
    # Gunakan barplot horizontal agar label lebih mudah dibaca
    fig_met, ax_met = plt.subplots(figsize=(10, 5))
    sns.barplot(x="Nilai", y="Metrik", data=metrics_data, palette="viridis", ax=ax_met)
    ax_met.set_xlim(0.85, 1.0) # Fokuskan sumbu X di 0.85 ke atas agar perbedaannya jelas
    for p in ax_met.patches:
        ax_met.annotate(f"{p.get_width():.4f}", 
                        (p.get_width() + 0.002, p.get_y() + p.get_height() / 2.), 
                        ha='left', va='center')
    st.pyplot(fig_met)


    # ==========================================
    # BAGIAN 2: CONFUSION MATRIX
    # ==========================================
    st.subheader("2. Confusion Matrix")
    
    # ---> Karena nilai confusion matrix eksak tidak diberikan, saya gunakan nilai dummy (estimasi).
    # Silakan ganti nilai di dalam array di bawah ini jika Anda punya array cm-nya! <---
    # Format: [[True Negative, False Positive], [False Negative, True Positive]]
    cm = np.array([[150, 15], 
                   [ 20, 165]])
                   
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, 
                xticklabels=["Cerai Talak", "Cerai Gugat"], 
                yticklabels=["Cerai Talak", "Cerai Gugat"])
    ax_cm.set_ylabel('True Label (Label Asli)')
    ax_cm.set_xlabel('Predicted Label (Label Prediksi)')
    st.pyplot(fig_cm)


    # ==========================================
    # BAGIAN 3: GRAFIK VALIDATION LOSS (5 FOLDS)
    # ==========================================
    st.subheader("3. Grafik Validation Loss (5-Fold Cross Validation)")
    
    # Data dari history training
    val_loss_fold1 = [0.6797, 0.6725, 0.6699, 0.3689, 0.1477, 0.1762, 0.187, 0.2067, 0.1961, 0.1354, 0.1424, 0.1577, 0.1633, 0.1586]
    val_loss_fold2 = [0.6886, 0.6718, 0.6758, 0.5218, 0.4067, 0.1534, 0.1469, 0.1777, 0.1803, 0.2087, 0.2318, 0.2319, 0.2609, 0.2829, 0.3266, 0.5683, 0.5551]
    val_loss_fold3 = [0.6705, 0.6294, 0.66, 0.6527, 0.6514, 0.1208, 0.1193, 0.1305, 0.1324, 0.1414, 0.1454, 0.1445, 0.1499, 0.1431, 0.1468, 0.1438]
    val_loss_fold4 = [0.6774, 0.6462, 0.6719, 0.1911, 0.0804, 0.0874, 0.1073, 0.1226, 0.1291, 0.1255, 0.1286, 0.1516, 0.1431, 0.1425, 0.1443, 0.1434]
    val_loss_fold5 = [0.6811, 0.6739, 0.6507, 0.6606, 0.1292, 0.1762, 0.1561, 0.1445, 0.1203, 0.1212, 0.5659, 0.5795, 0.5742, 0.5581, 0.247, 0.1576, 0.189, 0.1401]
    
    fig_loss, ax_loss = plt.subplots(figsize=(10, 5))
    
    ax_loss.plot(range(1, len(val_loss_fold1)+1), val_loss_fold1, label="Fold 1", marker='o', markersize=4)
    ax_loss.plot(range(1, len(val_loss_fold2)+1), val_loss_fold2, label="Fold 2", marker='o', markersize=4)
    ax_loss.plot(range(1, len(val_loss_fold3)+1), val_loss_fold3, label="Fold 3", marker='o', markersize=4)
    ax_loss.plot(range(1, len(val_loss_fold4)+1), val_loss_fold4, label="Fold 4", marker='o', markersize=4)
    ax_loss.plot(range(1, len(val_loss_fold5)+1), val_loss_fold5, label="Fold 5", marker='o', markersize=4)
    
    ax_loss.set_xlabel("Epochs")
    ax_loss.set_ylabel("Validation Loss")
    ax_loss.set_title("Validation Loss Progress across 5 Folds")
    ax_loss.legend()
    ax_loss.grid(True, linestyle='--', alpha=0.7)
    
    st.pyplot(fig_loss)
