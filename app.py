import os
import random
import math

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
import joblib

import cv2  
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.utils import to_categorical

# ==================== KONFIGURASI UMUM ====================

DATA_DIR = "dataset"       # folder dataset
MODEL_DIR = "models"       # folder simpan model
IMG_SIZE = (64, 64)        # ukuran citra setelah resize
AUG_PER_IMAGE = 20         # berapa banyak augmentasi per gambar asli

# Threshold untuk binarisasi setelah normalisasi (0–1)
THRESH_VALUE = 0.5         # bisa kamu ganti misal 0.4 atau 0.6

os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# =========================================================
#  PREPROCESSING CITRA (PROSES 1) & EKSTRAKSI FITUR (PROSES 2)
# =========================================================

def preprocessing_citra(pil_rgb, with_rotation=False):
    """
    Proses 1 – Preprocessing Citra (sesuai slide):
    - Resize
    - Grayscale
    - Gaussian Filtering
    - Rotasi (opsional, untuk augmentasi)
    - Thresholding

    Output:
    - pil_rgb_proc : RGB hasil resize/rotasi (PIL)
    - img_gray_proc: grayscale+Gaussian+threshold (PIL) → untuk ditampilkan
    - mask_bin     : array biner (0/1) → untuk fitur bentuk
    """
    # Resize
    pil_rgb = pil_rgb.resize(IMG_SIZE)

    # Rotasi kecil (kalau untuk augmentasi)
    if with_rotation:
        angle = random.uniform(-20, 20)
        pil_rgb = pil_rgb.rotate(angle)

    # Grayscale
    img_gray = pil_rgb.convert("L")

    # Gaussian Filtering
    img_gray_blur = img_gray.filter(ImageFilter.GaussianBlur(radius=1))

    # Normalisasi dan thresholding
    arr = np.array(img_gray_blur, dtype=np.float32) / 255.0
    mask_bin = (arr > THRESH_VALUE).astype(np.float32)

    img_gray_proc = Image.fromarray((mask_bin * 255).astype("uint8"))
    return pil_rgb, img_gray_proc, mask_bin


def ekstraksi_fitur_warna_hsv(pil_rgb):
    """
    Proses 2 – Ekstraksi Fitur Warna HSV:
    - Mean & std untuk H, S, V  → 6 fitur
    """
    arr_rgb = np.array(pil_rgb.convert("RGB"))
    hsv = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2HSV)

    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    fitur = [
        np.mean(h), np.std(h),
        np.mean(s), np.std(s),
        np.mean(v), np.std(v),
    ]
    return np.array(fitur, dtype=np.float32)


def ekstraksi_fitur_bentuk(mask_bin):
    """
    Ekstraksi fitur bentuk:
    1. Eccentricity
    2. Compactness   -> area / (width*height)
    3. Circularity   -> 4πA / P^2
    4. Aspect Ratio  -> width / height
    """
    mask_uint8 = (mask_bin * 255).astype("uint8")
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return np.zeros(4, dtype=np.float32)

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h if h > 0 else 0.0

    # compactness = seberapa penuh area terhadap bounding box
    compactness = float(area) / (w * h) if w * h > 0 else 0.0

    # circularity klasik
    circularity = (4.0 * math.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0

    # eccentricity pakai eigen value kovarian koordinat piksel
    pts = cnt.reshape(-1, 2).astype(np.float32)
    if pts.shape[0] >= 5:
        cov = np.cov(pts.T)
        eigvals, _ = np.linalg.eig(cov)
        eigvals = np.sort(eigvals)[::-1]  # besar → kecil
        if eigvals[0] > 0:
            eccentricity = float(np.sqrt(1 - eigvals[1] / eigvals[0]))
        else:
            eccentricity = 0.0
    else:
        eccentricity = 0.0

    return np.array([eccentricity, compactness, circularity, aspect_ratio],
                    dtype=np.float32)


def ekstraksi_fitur_gabungan(pil_rgb, with_rotation=False):
    """
    Pipeline lengkap:
    Preprocessing Citra → Ekstraksi Fitur Warna HSV → Ekstraksi Fitur Bentuk
    Output:
    - pil_rgb_proc, img_gray_proc, fitur (shape = (10,))
    """
    pil_rgb_proc, img_gray_proc, mask_bin = preprocessing_citra(
        pil_rgb, with_rotation=with_rotation
    )
    fitur_warna = ekstraksi_fitur_warna_hsv(pil_rgb_proc)
    fitur_bentuk = ekstraksi_fitur_bentuk(mask_bin)
    fitur = np.concatenate([fitur_warna, fitur_bentuk], axis=0)  # 10 fitur
    return pil_rgb_proc, img_gray_proc, fitur


def random_augment(pil_rgb):
    """
    Augmentasi sederhana:
    - flip horizontal/vertical
    - rotasi kecil
    - perubahan brightness & contrast
    Kemudian tetap masuk pipeline preprocessing+ekstraksi fitur.
    """
    img = pil_rgb
    if random.random() < 0.5:
        img = ImageOps.mirror(img)
    if random.random() < 0.5:
        img = ImageOps.flip(img)

    angle = random.uniform(-20, 20)
    img = img.rotate(angle)

    enhancer_b = ImageEnhance.Brightness(img)
    img = enhancer_b.enhance(random.uniform(0.7, 1.3))

    enhancer_c = ImageEnhance.Contrast(img)
    img = enhancer_c.enhance(random.uniform(0.7, 1.3))

    return img

# ==================== MODEL ====================

def build_bpnn(input_dim, num_classes):
    """Backpropagation Neural Network (MLP) berbasis vektor fitur."""
    model = Sequential([
        Dense(64, activation="relu", input_shape=(input_dim,)),
        Dense(32, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def build_lstm(input_shape, num_classes):
    """
    LSTM berbasis vektor fitur.
    Vektor fitur (panjang F) dipandang sebagai deret waktu dengan F timestep dan 1 fitur per timestep:
    input_shape = (timesteps=F, features=1)
    """
    model = Sequential([
        LSTM(64, input_shape=input_shape),
        Dense(32, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ==================== PERSIAPAN DATASET ====================

def prepare_dataset(data_dir, aug_per_image=AUG_PER_IMAGE):
    """
    Load dataset dari folder: data_dir/class_name/*.jpg

    Sesuai flowchart:
    - Upload banyak gambar
    - Preprocessing Citra
    - Ekstraksi Fitur (HSV + bentuk)
    - Hasil: vektor fitur untuk SVM, BPNN, LSTM
    """
    fitur_list = []
    labels = []

    class_names = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    if not class_names:
        raise ValueError(
            f"Tidak ditemukan subfolder kelas di '{data_dir}'. "
            "Pastikan struktur folder dataset sudah benar."
        )

    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        for fname in os.listdir(class_dir):
            fpath = os.path.join(class_dir, fname)
            if not os.path.isfile(fpath):
                continue

            try:
                pil_rgb = Image.open(fpath).convert("RGB")
            except Exception:
                continue

            # ---- gambar asli ----
            _, _, fitur = ekstraksi_fitur_gabungan(pil_rgb, with_rotation=False)
            fitur_list.append(fitur)
            labels.append(class_name)

            # ---- augmentasi ----
            for _ in range(aug_per_image):
                aug_rgb = random_augment(pil_rgb)
                _, _, fitur_aug = ekstraksi_fitur_gabungan(
                    aug_rgb, with_rotation=False
                )
                fitur_list.append(fitur_aug)
                labels.append(class_name)

    X = np.array(fitur_list)        # (N, F)
    labels = np.array(labels)
    # --- Scaling fitur (sangat penting untuk BPNN & SVM) ---
    scaler = StandardScaler()
    X = scaler.fit_transform(X)


    # LSTM: (N, timesteps, features_per_step) = (N, F, 1)
    X_seq = X.reshape(len(X), X.shape[1], 1)

    le = LabelEncoder()
    y_int = le.fit_transform(labels)
    num_classes = len(le.classes_)
    y_onehot = to_categorical(y_int, num_classes=num_classes)

    X_train, X_test, X_seq_train, X_seq_test, \
    y_int_train, y_int_test, y_onehot_train, y_onehot_test = train_test_split(
        X, X_seq, y_int, y_onehot,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_int,
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "X_seq_train": X_seq_train,
        "X_seq_test": X_seq_test,
        "y_int_train": y_int_train,
        "y_int_test": y_int_test,
        "y_onehot_train": y_onehot_train,
        "y_onehot_test": y_onehot_test,
        "label_encoder": le,
        "class_names": list(le.classes_),
        "n_features": X.shape[1],
        "scaler": scaler,
    }

def save_models(svm_model, bpnn_model, lstm_model, label_encoder, scaler):
    joblib.dump(svm_model, os.path.join(MODEL_DIR, "svm_model.pkl"))
    bpnn_model.save(os.path.join(MODEL_DIR, "bpnn_model.h5"))
    lstm_model.save(os.path.join(MODEL_DIR, "lstm_model.h5"))
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

def load_models_if_exist():
    paths = {
        "svm": os.path.join(MODEL_DIR, "svm_model.pkl"),
        "bpnn": os.path.join(MODEL_DIR, "bpnn_model.h5"),
        "lstm": os.path.join(MODEL_DIR, "lstm_model.h5"),
        "le": os.path.join(MODEL_DIR, "label_encoder.pkl"),
        "scaler": os.path.join(MODEL_DIR, "scaler.pkl"),
    }
    if not all(os.path.exists(p) for p in paths.values()):
        return None, None, None, None, None

    svm_model = joblib.load(paths["svm"])
    bpnn_model = load_model(paths["bpnn"])
    lstm_model = load_model(paths["lstm"])
    label_encoder = joblib.load(paths["le"])
    scaler = joblib.load(paths["scaler"])
    return svm_model, bpnn_model, lstm_model, label_encoder, scaler

def preprocess_uploaded_image(uploaded_file):
    """
    Preprocessing & Ekstraksi Fitur untuk 1 gambar (Testing):
    - Resize, grayscale, Gaussian, threshold
    - Ekstraksi fitur HSV + bentuk
    """
    pil_rgb = Image.open(uploaded_file).convert("RGB")
    pil_rgb_proc, img_gray_proc, fitur = ekstraksi_fitur_gabungan(
        pil_rgb, with_rotation=False
    )

    X_raw = fitur.reshape(1, -1)
    return pil_rgb_proc, img_gray_proc, X_raw

# ==================== KONFIGURASI STREAMLIT ====================

st.set_page_config(
    page_title="Klasifikasi Rumput Laut",
    page_icon="🌊",
    layout="wide",
)

st.sidebar.title("Menu")
menu = st.sidebar.radio(
    "Pilih halaman",
    [
        "Home",
        "Training (Upload banyak gambar)",
        "Validasi Model",
        "Testing (Upload 1 gambar)",
        "Tentang Dataset",
    ],
)

st.sidebar.info(
    "Aplikasi klasifikasi rumput laut\n"
    "Proses: Preprocessing Citra → Ekstraksi Fitur → Klasifikasi\n"
    "Model: SVM, BPNN, dan LSTM."
)

# ==================== HALAMAN: HOME ====================

if menu == "Home":
    st.title("🌊 Klasifikasi Rumput Laut dengan SVM, BPNN, dan LSTM")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Proses 1 – Preprocessing Citra")
        st.markdown("""
        Tahapan yang dilakukan pada setiap citra:
        1. **Resize** ke ukuran tetap `64×64` piksel  
        2. **Grayscale**  
        3. **Gaussian Filtering** (mengurangi noise)  
        4. **Rotasi** kecil (digunakan saat augmentasi data)  
        5. **Thresholding** → citra biner (0 dan 1)
        """)

    with col2:
        st.subheader("Proses 2 – Ekstraksi Fitur")
        st.markdown("""
        - **Fitur Warna HSV**  
          Mean dan standar deviasi untuk masing-masing kanal **Hue, Saturation, Value** (6 fitur).  
        - **Fitur Bentuk**  
          1. *Eccentricity*  
          2. *Compactness*  
          3. *Circularity*  
          4. *Aspect Ratio*  
          
        Total fitur yang digunakan: **10 fitur per citra**.
        """)

    st.markdown("---")
    st.subheader("Alur Sistem (disesuaikan dengan Flowchart)")
    st.markdown("""
    1. **Training – Upload banyak gambar**  
       - Dataset citra dimuat dari folder `dataset/kelas/`.  
       - Setiap citra → *Preprocessing Citra* → *Ekstraksi Fitur*.  
       - Fitur dipakai untuk melatih **SVM**, **BPNN**, dan **LSTM**.  
       - Model disimpan dalam folder `models/`.

    2. **Testing – Upload 1 gambar**  
       - Pengguna meng-upload satu citra.  
       - Citra melalui *Preprocessing Citra* dan *Ekstraksi Fitur*.  
       - Ketiga model memberikan prediksi kelas dan probabilitas (%).  
    """)

# ==================== HALAMAN: TRAINING ====================

elif menu == "Training (Upload banyak gambar)":
    st.title("🔧 Training Model (Upload banyak gambar)")
    st.write("""
    Tahap ini mewakili alur **\"Upload banyak gambar → Preprocessing dan Ekstraksi Fitur → Prediksi 3 model\"** pada flowchart.
    """)

    if st.button("Mulai Training"):
        with st.spinner("Menyiapkan dataset (preprocessing & ekstraksi fitur)..."):
            data = prepare_dataset(DATA_DIR)

        st.success(
            "Dataset siap! Jumlah sampel setelah augmentasi: "
            f"{len(data['X_train']) + len(data['X_test'])}"
        )

        # ---- SVM ----
        st.subheader("Training SVM (berbasis fitur)")
        svm_model = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)
        svm_model.fit(data["X_train"], data["y_int_train"])
        y_pred_svm = svm_model.predict(data["X_test"])
        acc_svm = accuracy_score(data["y_int_test"], y_pred_svm)
        st.write(f"✅ Akurasi SVM (test): **{acc_svm*100:.2f}%**")

        # ---- BPNN ----
        st.subheader("Training BPNN (MLP, berbasis fitur)")
        input_dim = data["X_train"].shape[1]
        num_classes = len(data["class_names"])
        bpnn_model = build_bpnn(input_dim, num_classes)
        bpnn_model.fit(
            data["X_train"], data["y_onehot_train"],
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            verbose=0,
        )
        loss_bp, acc_bp = bpnn_model.evaluate(
            data["X_test"], data["y_onehot_test"], verbose=0
        )
        st.write(f"✅ Akurasi BPNN (test): **{acc_bp*100:.2f}%**")

        # ---- LSTM ----
        st.subheader("Training LSTM (deret fitur)")
        input_shape = (
            data["X_seq_train"].shape[1],
            data["X_seq_train"].shape[2],
        )
        lstm_model = build_lstm(input_shape, num_classes)
        lstm_model.fit(
            data["X_seq_train"], data["y_onehot_train"],
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            verbose=0,
        )
        loss_ls, acc_ls = lstm_model.evaluate(
            data["X_seq_test"], data["y_onehot_test"], verbose=0
        )
        st.write(f"✅ Akurasi LSTM (test): **{acc_ls*100:.2f}%**")

        save_models(
            svm_model,
            bpnn_model,
            lstm_model,
            data["label_encoder"],
            data["scaler"],
        )
        st.success(
            "Semua model & label encoder berhasil disimpan di folder 'models/'."
        )

        st.session_state["train_data"] = data

# ==================== HALAMAN: VALIDASI ====================

elif menu == "Validasi Model":
    st.title("📊 Validasi / Evaluasi Model")

    svm_model, bpnn_model, lstm_model, label_encoder, scaler = load_models_if_exist()
    if svm_model is None:
        st.warning("Model belum ditemukan. Silakan lakukan training terlebih dahulu.")
    else:
        if "train_data" in st.session_state:
            data = st.session_state["train_data"]
        else:
            data = prepare_dataset(DATA_DIR)

        class_names = data["class_names"]

        st.write(
            "Performa model pada **data test** (ditampilkan sebagai grafik, dalam persen %)."
        )

        # ================== TABEL & GRAFIK PREC/REC/F1 ==================
        def plot_report(y_true, y_pred, class_names, title):
            report_dict = classification_report(
                y_true, y_pred, target_names=class_names, output_dict=True
            )
            df = pd.DataFrame(report_dict).T

            for col in ["precision", "recall", "f1-score"]:
                if col in df.columns:
                    df[col] = df[col] * 100.0

            acc = report_dict["accuracy"] * 100.0

            metric_cols = ["precision", "recall", "f1-score"]
            df_class = df.loc[class_names, metric_cols].round(2)
            df_avg = df.loc[["macro avg", "weighted avg"], metric_cols].round(2)

            rename_map = {
                "precision": "precision (%)",
                "recall": "recall (%)",
                "f1-score": "f1-score (%)",
            }
            df_class = df_class.rename(columns=rename_map)
            df_avg = df_avg.rename(columns=rename_map)

            st.subheader(title)
            st.write(f"Akurasi: **{acc:.2f}%**")
            st.markdown("**Per kelas**")
            st.bar_chart(df_class)
            st.markdown("**Rata-rata (macro & weighted)**")
            st.bar_chart(df_avg)

        # ================== HEATMAP CONFUSION MATRIX ==================
        def plot_confusion_matrix_heatmap(y_true, y_pred, class_names, title):
            cm = confusion_matrix(y_true, y_pred)

            # normalisasi per baris untuk colorbar 0–1
            cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
            cm_norm = np.nan_to_num(cm_norm)

            fig, ax = plt.subplots(figsize=(8, 4))
            sns.heatmap(
                cm_norm,
                annot=cm,               # angka yang muncul = jumlah sampel
                fmt="d",
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={"label": "Proporsi"},
                ax=ax,
            )
            ax.set_xlabel("Predicted label")
            ax.set_ylabel("True label")
            ax.set_title(title)
            plt.yticks(rotation=0)
            st.pyplot(fig)

        # ================== HITUNG PREDIKSI ==================
        # SVM
        y_pred_svm = svm_model.predict(data["X_test"])
        plot_report(data["y_int_test"], y_pred_svm, class_names, "SVM (berbasis fitur)")

        # BPNN
        y_proba_bp = bpnn_model.predict(data["X_test"], verbose=0)
        y_pred_bp = np.argmax(y_proba_bp, axis=1)
        plot_report(
            data["y_int_test"], y_pred_bp, class_names, "BPNN (MLP berbasis fitur)"
        )

        # LSTM
        y_proba_ls = lstm_model.predict(data["X_seq_test"], verbose=0)
        y_pred_ls = np.argmax(y_proba_ls, axis=1)
        plot_report(
            data["y_int_test"], y_pred_ls, class_names, "LSTM (berbasis deret fitur)"
        )

        # ============= BAGIAN YANG DIGANTI: CONFUSION MATRIX =============
        with st.expander("Lihat Confusion Matrix"):
            st.markdown("#### SVM")
            plot_confusion_matrix_heatmap(
                data["y_int_test"], y_pred_svm, class_names,
                "Confusion Matrix – SVM"
            )

            st.markdown("#### BPNN")
            plot_confusion_matrix_heatmap(
                data["y_int_test"], y_pred_bp, class_names,
                "Confusion Matrix – BPNN"
            )

            st.markdown("#### LSTM")
            plot_confusion_matrix_heatmap(
                data["y_int_test"], y_pred_ls, class_names,
                "Confusion Matrix – LSTM"
            )

# ==================== HALAMAN: TESTING ==================== #

elif menu == "Testing (Upload 1 gambar)":
    st.title("🧪 Testing Gambar Baru (Upload 1 gambar)")
    st.write("""
    Tahap ini mewakili alur **\"Upload 1 gambar → Preprocessing dan Ekstraksi Fitur → Prediksi 3 model\"** pada flowchart.
    """)

    svm_model, bpnn_model, lstm_model, label_encoder, scaler = load_models_if_exist()
    if svm_model is None:
        st.warning("Model belum ditemukan. Silakan lakukan training terlebih dahulu.")
    else:
        uploaded_file = st.file_uploader(
            "Upload gambar rumput laut (jpg/png)",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded_file is not None:
            pil_rgb_proc, img_gray_proc, X_raw = preprocess_uploaded_image(
                uploaded_file
            )

            # gunakan scaler yang sama dengan saat training
            X = scaler.transform(X_raw)
            X_seq = X.reshape(1, X.shape[1], 1)

            col1, col2 = st.columns(2)
            with col1:
                st.image(
                    pil_rgb_proc,
                    caption="Citra setelah Resize (RGB)",
                    use_column_width=True,
                )
                st.image(
                    img_gray_proc,
                    caption="Hasil Preprocessing (Grayscale + Gaussian + Thresholding)",
                    use_column_width=True,
                )

            class_names = label_encoder.classes_

            # SVM
            pred_svm = svm_model.predict(X)[0]
            prob_svm = svm_model.predict_proba(X)[0]
            label_svm = label_encoder.inverse_transform([pred_svm])[0]
            max_prob_svm = float(np.max(prob_svm)) * 100.0

            # BPNN
            proba_bp = bpnn_model.predict(X, verbose=0)[0]
            idx_bp = np.argmax(proba_bp)
            label_bp = class_names[idx_bp]
            max_prob_bp = float(np.max(proba_bp)) * 100.0

            # LSTM
            proba_ls = lstm_model.predict(X_seq, verbose=0)[0]
            idx_ls = np.argmax(proba_ls)
            label_ls = class_names[idx_ls]
            max_prob_ls = float(np.max(proba_ls)) * 100.0

            with col2:
                st.markdown("### Hasil Prediksi (Deteksi kategori)")
                st.write(
                    f"**SVM** memprediksi: `{label_svm}` (prob: {max_prob_svm:.2f}%)"
                )
                st.write(
                    f"**BPNN** memprediksi: `{label_bp}` (prob: {max_prob_bp:.2f}%)"
                )
                st.write(
                    f"**LSTM** memprediksi: `{label_ls}` (prob: {max_prob_ls:.2f}%)"
                )

                st.markdown("#### Probabilitas per kelas (LSTM)")
                prob_table = {
                    "Kelas": class_names,
                    "Probabilitas (%)": [
                        round(float(p) * 100.0, 2) for p in proba_ls
                    ],
                }
                st.dataframe(prob_table)

# ==================== HALAMAN: TENTANG DATASET ====================

elif menu == "Tentang Dataset":
    st.title("📁 Tentang Dataset Rumput Laut")
    st.write("""
    Dataset berisi citra beberapa jenis rumput laut, misalnya:
    
    1. *Caulerpa lentillifera*
    2. *Acanthophora spicifera*
    3. *Cetraria muricata*
    4. *Codium taylorii*
    5. *Gigartina pistillata*
    6. *Gracilaria salicornia*
    7. *Hypnea musciformis*
    8. *Padina australis*
    9. *Sargassum scabridum*
    
    Semakin banyak citra asli per kelas (bukan hanya augmentasi), 
    semakin baik performa model.
    """)

    if os.path.exists(DATA_DIR):
        st.subheader("Ringkasan isi folder dataset/")
        for class_name in sorted(os.listdir(DATA_DIR)):
            class_dir = os.path.join(DATA_DIR, class_name)
            if os.path.isdir(class_dir):
                n_files = sum(
                    1
                    for f in os.listdir(class_dir)
                    if os.path.isfile(os.path.join(class_dir, f))
                )
                st.write(f"- {class_name}: {n_files} file gambar")
    else:
        st.warning("Folder dataset/ belum ada. Buat terlebih dahulu.")
