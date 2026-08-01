import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained Spam/Ham model
model = joblib.load(r"C:\Users\Dell\OneDrive\Desktop\Spam Email\spam_detection.pkl")  # Change filename if needed

# Mapping for predictions
label_map_num = {0: "📨 Ham (Not Spam)", 1: "🚨 Spam"}
label_map_str = {"ham": "📨 Ham (Not Spam)", "spam": "🚨 Spam"}

# Page configuration
st.set_page_config(
    page_title="Spam/Ham Classification",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
.main {
    background-color: #f8fafc;
    padding: 20px;
    font-family: 'Roboto', sans-serif;
}
.stButton>button {
    background: linear-gradient(90deg, #2563eb, #1d4ed8);
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
    border: none;
    font-weight: 600;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #1d4ed8, #1e3a8a);
    transform: scale(1.03);
}
.stTextArea>div>textarea {
    border-radius: 8px;
    padding: 12px;
    border: 1px solid #d1d5db;
    background-color: #ffffff;
    font-size: 15px;
}
.stFileUploader>div {
    border-radius: 8px;
    border: 1px solid #d1d5db;
    background-color: #ffffff;
}
.card {
    background-color: white;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.08);
    margin-bottom: 20px;
    transition: all 0.3s ease;
}
.card:hover {
    box-shadow: 0 5px 16px rgba(0,0,0,0.12);
}
.header {
    background: linear-gradient(90deg, #2563eb, #1e3a8a);
    color: white;
    padding: 20px;
    border-radius: 8px;
    text-align: center;
    margin-bottom: 25px;
}
.stDataFrame {
    border-radius: 8px;
    overflow: hidden;
}
.sidebar .sidebar-content {
    background-color: #eff6ff;
    border-right: 1px solid #d1d5db;
}
.sentiment-positive {
    color: #16a34a;
    font-weight: bold;
}
.sentiment-negative {
    color: #dc2626;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header">
    <h1>📧 Spam/Ham Classifier</h1>
    <p>Detect whether a message is spam or ham with accuracy and speed!</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    try:
        st.image(r"C:\Users\Dell\OneDrive\Desktop\Spam Email\spam_detect_img.png", caption="Spam/Ham Detection", use_container_width=True)
    except FileNotFoundError:
        st.warning("Image 'spam_ham_image.png' not found. Please add it or remove this line.")
    st.header("📞 Contact")
    st.markdown("""
    - 📧 Email: priyarao1604@gmail.com
    - 📱 Phone: +91 8708316414
    - 🌐 LinkedIn: www.linkedin.com/in/priyarao2
    """)
    
    st.header("ℹ️ About")
    st.markdown("""
    Created by Priya, passionate about building AI solutions for text classification.
    """)

# Main content
col1, col2 = st.columns([3, 2])

# -------- Single Message Analysis --------
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📝 Single Message Analysis")

    message_input = st.text_area(
        "Enter Message",
        placeholder="Type your message here (e.g., 'Win a free iPhone now!' or 'Hey, let's catch up tomorrow.')",
        height=150,
        help="Enter a single message to check if it is spam or ham."
    )

    if st.button("Classify Message", key="analyze_single"):
        if message_input.strip():
            try:
                prediction = model.predict([message_input])[0]
                confidence = model.predict_proba([message_input]).max()

                if isinstance(prediction, str):
                    label = label_map_str.get(prediction.lower(), "Unknown")
                    css_class = "sentiment-positive" if prediction.lower() == "ham" else "sentiment-negative"
                else:
                    label = label_map_num.get(int(prediction), "Unknown")
                    css_class = "sentiment-positive" if int(prediction) == 0 else "sentiment-negative"

                st.markdown(f"""
                <p>Prediction: <span class="{css_class}">{label}</span></p>
                <p>Confidence: {confidence:.2%}</p>
                """, unsafe_allow_html=True)

                if (isinstance(prediction, str) and prediction.lower() == "ham") or (prediction == 0):
                    st.balloons()

            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a message.")
    st.markdown('</div>', unsafe_allow_html=True)

# -------- Bulk Message Analysis --------
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📂 Bulk Message Analysis")

    uploaded_file = st.file_uploader(
        "Upload CSV, TXT, or Excel File",
        type=['csv', 'txt', 'xls', 'xlsx'],
        help="Upload a file with messages in a single column. Column name should be 'message' for CSV/Excel or plain lines for TXT."
    )

    if uploaded_file is not None:
        try:
            # Detect file type
            file_extension = uploaded_file.name.split('.')[-1].lower()

            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension == 'txt':
                df = pd.read_csv(uploaded_file, names=['message'], sep='\t')
            elif file_extension in ['xls', 'xlsx']:
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format.")
                df = None

            if df is not None:
                if 'message' not in df.columns:
                    st.error("The file must contain a 'message' column.")
                else:
                    st.dataframe(df, use_container_width=True)

                    if st.button("Analyze Messages", key="analyze_bulk"):
                        predictions = model.predict(df['message'])
                        confidences = model.predict_proba(df['message']).max(axis=1)

                        final_labels = []
                        for pred in predictions:
                            if isinstance(pred, str):
                                final_labels.append(label_map_str.get(pred.lower(), "Unknown"))
                            else:
                                final_labels.append(label_map_num.get(int(pred), "Unknown"))

                        df['Prediction'] = final_labels
                        df['Confidence'] = [f"{c:.2%}" for c in confidences]
                        st.dataframe(df, use_container_width=True)

                        csv_download = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv_download,
                            file_name="spam_ham_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 25px; color: #4b5563; font-size: 0.9em;">
    <p>Developed by Priya | © 2026 Spam/Ham Classifier</p>
</div>
""", unsafe_allow_html=True)
