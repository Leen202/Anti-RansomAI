import streamlit as st
import joblib
import os
import numpy as np
from PyPDF2 import PdfReader
import docx

# ✅ تدريب النموذج تلقائيًا إذا ما كان موجود
if not os.path.exists("ransomware_model.joblib"):
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    np.random.seed(42)
    data = {
        'file_size_kb': np.random.randint(10, 10000, 200),
        'num_words': np.random.randint(10, 10000, 200),
        'file_extension': np.random.choice(['.exe', '.txt', '.docx', '.pdf'], 200),
        'label': np.random.choice(['benign', 'ransomware'], 200)
    }
    df = pd.DataFrame(data)
    df = pd.get_dummies(df, columns=['file_extension'])
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['label'])
    X = df.drop(columns=['label', 'label_encoded'])
    y = df['label_encoded']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, "ransomware_model.joblib")
    joblib.dump(le, "label_encoder.joblib")

# ✅ واجهة التطبيق
st.title("🔐 Anti-Ransom AI - File Behavior Analyzer")

st.write("""
Upload a file (TXT, PDF, DOCX, EXE) to check if it behaves like ransomware.
The model analyzes file size, number of words, and extension.
""")

uploaded_file = st.file_uploader("Choose a file", type=['txt', 'pdf', 'docx', 'exe'])

if uploaded_file:
    try:
        content = uploaded_file.read()
        file_size_kb = len(content) // 1024
        ext = os.path.splitext(uploaded_file.name)[1]
        num_words = 0

        if ext == ".txt":
            num_words = len(content.decode(errors='ignore').split())
        elif ext == ".pdf":
            try:
                uploaded_file.seek(0)
                reader = PdfReader(uploaded_file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                num_words = len(text.split())
            except:
                num_words = 0
        elif ext == ".docx":
            try:
                uploaded_file.seek(0)
                doc = docx.Document(uploaded_file)
                text = "\n".join([para.text for para in doc.paragraphs])
                num_words = len(text.split())
            except:
                num_words = 0

        extensions = ['.docx', '.exe', '.pdf', '.txt']
        ext_encoded = [1 if ext == e else 0 for e in extensions]
        features = [file_size_kb, num_words] + ext_encoded
        X = np.array([features])

        model = joblib.load("ransomware_model.joblib")
        encoder = joblib.load("label_encoder.joblib")
        pred = model.predict(X)[0]
        label = encoder.inverse_transform([pred])[0]

        st.subheader("🧠 Prediction Result:")
        if label == "ransomware":
            st.error("⚠️ Detected: Ransomware-like behavior!")
        else:
            st.success("✅ File appears to be safe.")
    except Exception as e:
        st.error("Something went wrong while processing the file.")
        st.text(str(e))
