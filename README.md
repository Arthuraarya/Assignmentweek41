 # Assignmentweek41
Streamlit dashboard for bank churn.
📊 Bank Churn Dashboard (Streamlit)
Dashboard interaktif untuk analisis churn nasabah: eksplorasi data, pemodelan prediksi, dan interpretasi faktor yang memengaruhi churn. Dibangun dengan Python + Streamlit dan siap dideploy ke Streamlit Community Cloud.

🎯 Tujuan Proyek
Memahami profil nasabah dan perilaku yang berkaitan dengan churn (berhenti menjadi pelanggan).
Memprediksi risiko churn menggunakan model ML sederhana (Logistic Regression / Random Forest).
Menjelaskan driver churn dengan Permutation Importance dan Partial Dependence agar bisa ditindaklanjuti (retensi, cross-sell, dsb).

🗂️ Tentang Data (bank_churn_data.csv)

Dataset berisi catatan nasabah dan status churn.
Target: attrition_flag (diubah ke kolom biner churn: 1=Attrited/Churn, 0=Existing).
Kolom penting yang digunakan (contoh umum, bisa berbeda tergantung datasetmu):
Demografi: customer_age, gender, education_level, income_category, marital_status, dependent_count
Perilaku & transaksi: total_trans_amt, total_trans_ct, total_revolving_bal, credit_limit
Lama berlangganan & interaksi: months_on_book, months_inactive_12_mon, contacts_count_12_mon
Lainnya: card_category

Catatan:

File harus bernama bank_churn_data.csv dan diletakkan di root repo.
Kelas churn biasanya imbalance (~10–20% churn). Model dan metrik mempertimbangkan hal ini (pakai class_weight="balanced", ROC/PR AUC).

🧪 Apa yang Dilakukan Aplikasi

Aplikasi memiliki 3 halaman (via sidebar):
EDA
Histogram variabel numerik (bisa pilih kolom).
Churn rate per kategori (pilih kolom kategorikal).
Distribusi usia & churn rate per kelompok usia.

Modeling
Pilih model: Logistic Regression / Random Forest.
Train–test split terstratifikasi.
Evaluasi: Accuracy, Precision, Recall, F1, ROC AUC, PR AUC.
Kurva ROC dan Precision–Recall (Plotly interaktif).

Interpretasi

Permutation Feature Importance (pilih top-N via slider).
Partial Dependence Plot untuk fitur numerik teratas (bila tersedia).

⚠️ Catatan & Batasan

Dataset contoh, bukan data riil perusahaan; hasil tidak mewakili kondisi bisnis sebenarnya.
Fitur & kualitas prediksi bergantung pada kualitas/kelengkapan data.
Untuk produksi, pertimbangkan:
Validasi data real-time
Model monitoring & retraining
Penanganan privasi & kepatuhan data
