import streamlit as st
import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    f1_score,
)
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import seaborn as sns


# Fungsi untuk menampilkan confusion matrix menggunakan Matplotlib
def plot_confusion_matrix(cm, kernel):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    ax.set_title(f"Confusion Matrix {kernel}")
    st.pyplot(fig)


# Fungsi untuk memuat model
def load_model(file_path):
    with open(file_path, "rb") as file:
        model = pickle.load(file)
    return model


# Fungsi untuk menghasilkan data dummy
def load_data():
    # iris = datasets.load_iris()
    # df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    # df["target"] = iris.target
    df = pd.read_csv(
        "https://raw.githubusercontent.com/dewialqurani/Skripsi/main/data_pre.csv",
        sep=";",
    )
    return df


# Fungsi untuk preprocessing (contoh: split data)
def preprocess_data(df):
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    return X_train, X_test, y_train, y_test


# Fungsi untuk training model dan evaluasi
def train_and_evaluate(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro")
    rec = recall_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)
    return acc, prec, rec, cm, model


# Main program Streamlit
# def main():
with st.sidebar:
    tab_choice = option_menu(
        menu_title=None,
        options=["Dataset", "Preprocess", "Hasil Klasifikasi", "Implementasi"],
        default_index=0,
    )
st.title("Website Klasifikasi Status Stunting")

# Pilihan tab
# tab_choice = st.sidebar.selectbox(
#     "Pilih Tab", ["Dataset", "Preprocess", "Hasil", "Implementasi"]
# )

# Load dataset
# df = load_data()

# Tab Dataset
if tab_choice == "Dataset":
    df_dataset = pd.read_csv(
        "https://raw.githubusercontent.com/dewialqurani/Skripsi/main/data_pre.csv",
        sep=";",
    )
    st.write("Stunting merujuk pada kondisi dimana tinggi badan balita berada di bawah nilai rata-rata. Kondisi ini disebabkan oleh kurangnya asupan gizi yang berlangsung dalam jangka waktu yang signifikan")
    st.header("Dataset")
    st.write("""Data yang digunakan diperoleh dari Dinas Kesehatan Kabupaten Probolinggo yang berisi data UPT Puskesmas Tegalsiwalan Kabupaten Probolinggo pada bulan Agustus 2023 yang berjumlah 2100 data dengan rincian 300 data stunting dan 1800 data tidak stunting """)
    st.write("Dataframe:")
    st.write(df_dataset)

# Tab Preprocess
elif tab_choice == "Preprocess":
    st.header("Pre-processing Data")
    st.write("Prepocessing Data merupakan suatu tahapan dari data mining, dimana prepocessing dilakukan guna untuk mengolah suatu data awal atau data asli menjadi data atau inputan yang berkualitas sebelum dilanjutkan pada tahapan berikutnya.")

    st.header("One Hot Encoding")
    st.write("Untuk penanganan data kategorikal, dalam penelitian ini transformasi data dilakukan dengan menggunakan one-hot encoding dimana dapat mempresentasikan dta dengan tipe kategori sebagai vektor biner yang bernilai integer, 0 dan 1 mana dengan menggunakan one-hot encoding setiap value pada data kategori akan berdiri sendiri sebagai fitur dengan nilai 1 untuk kondisi benar dan 0 untuk kondisi salah")
    df_hapuskolom = pd.read_csv(
        "https://raw.githubusercontent.com/dewialqurani/Skripsi/main/DataHAPUSKOLOM.csv",
        sep=",",
    )
    st.write(df_hapuskolom)

    st.header("Transformasi Data")
    st.write("Dibagian ini terjadi proses perubahan pada data ke dalam bentuk atau format yang akan di proses oleh sistem, dimana nantinya akan memudahkan dalam pengelolaan data tersebut. Transformasi dilakukan pada kolom diagnosa dimana IYA bernilai 1 dan TIDAK bernilai 0")
    df_hasilencode = pd.read_csv(
        "https://raw.githubusercontent.com/dewialqurani/Skripsi/main/DataHasilEncode.csv",
        sep=",",
    )
    st.write(df_hasilencode)


# Tab Hasil
elif tab_choice == "Hasil Klasifikasi":
    label_mod = [
        "NoSMOTE",
        "withSMOTE",
        "withEuclidean",
        "withmanhattan",
        "Undersampling",
    ]
    header_name = [
        "Skenario Uji Coba SVM Tanpa SMOTE",
        "Skenario Uji Coba SVM dengan SMOTE",
        "Skenario Uji Coba SVM dengan SMOTE Euclidean",
        "Skenario Uji Coba SVM dengan SMOTE Manhattan",
        "Skenario Uji Coba SVM Undersampling",
    ]
    kernels = ["linear", "poly", "rbf", "sigmoid"]
    # df_prep = pd.read_csv(
    #     "https://raw.githubusercontent.com/dewialqurani/Skripsi/main/StuntingPre.csv",
    #     sep=",",
    # )
    df_prep = pd.read_excel("StuntingPre.xlsx")
    rand_state = [42, 0, 0, None, 0]
    # path_model = "E:\dewi\model"
    X = df_prep[["Usia Saat Ukur", "Berat", "Tinggi", "JK_L", "JK_P"]]  # Fitur (input)
    y = df_prep["Diagnosa"]  # Target (output)
    for i, label_ in enumerate(label_mod):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=rand_state[i]
        )
        if label_ == "Undersampling":
            from sklearn.preprocessing import StandardScaler
            from imblearn.under_sampling import RandomUnderSampler

            # Assuming X and y are defined
            # Split the dataset into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=0
            )

            # Display the class distribution in the training data before undersampling
            label_kelas = np.array(y_train)

            # Scaling the data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Initialize RandomUnderSampler
            undersampler = RandomUnderSampler(random_state=0)

            # Apply undersampling on the training data
            X_train_balanced, y_train_balanced = undersampler.fit_resample(
                X_train_scaled, y_train
            )
            # Apply undersampling on the test data
            X_test, y_test = undersampler.fit_resample(X_test_scaled, y_test)

        # Menampilkan confusion matrix untuk setiap kernel dalam satu baris
        st.header(f"{header_name[i]}")
        cols = st.columns(4)
        for j, kernel in enumerate(kernels):
            path_ = f"model/{label_}model_{kernel}.pkl"
            model = load_model(path_)
            y_pred_test = model.predict(X_test)
            accuracy_test = accuracy_score(y_test, y_pred_test)
            precision = precision_score(y_test, y_pred_test)
            recall = recall_score(y_test, y_pred_test)
            f1 = f1_score(y_test, y_pred_test)
            cm_test = confusion_matrix(y_test, y_pred_test)
            # results[kernel]['cm_test'] = cm_test
            report = classification_report(y_test, y_pred_test, zero_division=0)
            # st.write(f"akurasi {label_} {kernel} : ")
            # st.write(accuracy_test)
            # plot_confusion_matrix(cm_test, kernel)
            with cols[j]:
                st.write(f"Kernel - {kernel}")
                st.write(f"akurasi: {accuracy_test* 100:.2f}%")
                st.write(f"precision: {precision* 100:.2f}%")
                st.write(f"recall: {recall* 100:.2f}%")
                st.write(f"f1 score: {f1* 100:.2f}%")
                # st.write(report)
                fig = plot_confusion_matrix(cm_test, kernel)
    # st.header("Hasil Evaluasi Model")
    # X_train, X_test, y_train, y_test = preprocess_data(df)
    # acc, prec, rec, cm, model = train_and_evaluate(X_train, X_test, y_train, y_test)
    # st.write("Accuracy:", acc)
    # st.write("Precision:", prec)
    # st.write("Recall:", rec)
    # st.write("Confusion Matrix:")
    # st.write(cm)

# Tab Implementasi
elif tab_choice == "Implementasi":
    st.header("Implementasi Klasifikasi")
    st.write("Silakan masukkan nilai untuk klasifikasi:")
    metode = st.selectbox(
        "Pilih Metode SVM + :",
        (
            "Tanpa Menggunakan SMOTE",
            "Menggunakan SMOTE",
            "Menggunakan SMOTE Euclidean Distance",
            "Menggunakan SMOTE Manhattan Distance",
            "Menggunakan Undersampling",
        ),
    )
    pathkey = {
        "Tanpa Menggunakan SMOTE": "NoSMOTE",
        "Menggunakan SMOTE": "withSMOTE",
        "Menggunakan SMOTE Euclidean Distance": "withEuclidean",
        "Menggunakan SMOTE Manhattan Distance": "withmanhattan",
        "Menggunakan Undersampling": "Undersampling",
    }

    kernel = st.selectbox("Pilih Kernel:", ("linear", "rbf", "poly", "sigmoid"))
    umur = st.number_input("Umur", min_value=0)
    berat_badan = st.number_input("Berat Badan", min_value=0.0)
    tinggi_badan = st.number_input("Tinggi Badan", min_value=0)
    jk = st.selectbox("Pilih Jenis Kelamin:", ("Laki-Laki", "Perempuan"))
    if jk == "Laki-Laki":
        data_in = [umur, berat_badan, tinggi_badan, 1, 0]
    else:
        data_in = [umur, berat_badan, tinggi_badan, 1, 0]

    if st.button("Prediksi"):
        # modelpath = "model/bestmodel.pkl"
        modelpath = f"model/{pathkey[metode]}model_{kernel}.pkl"
        bModel = load_model(modelpath)
        scaler = load_model("model/scaler.pkl")

        st.write("Hasil:")
        st.write(f"SVM Model {metode} dengan kernel {kernel}")
        # data_in = [25, 70, 170, 1, 0]
        # Mengubah data tunggal menjadi array numpy 2D (matriks)
        data_in_2d = np.array(data_in).reshape(1, -1)

        # Normalisasi data menggunakan scaler yang telah dimuat
        data_normalized = scaler.transform(data_in_2d)
        # Prediksi menggunakan model yang sudah ditraining
        prediction = bModel.predict(data_normalized)
        # species = iris.target_names[prediction[0]]
        label_kelas = ["Normal", "Stunting"]
        if prediction[0] == 0:
            st.success(f"Prediksi : {label_kelas[prediction[0]]}")
        else:
            st.warning(f"Prediksi : {label_kelas[prediction[0]]}")


# if __name__ == "__main__":
#     main()
