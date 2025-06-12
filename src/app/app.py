import streamlit as st
import torch
from PIL import Image
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cnn_model import PneumoniaCNN, load_model, predict_image

st.set_page_config(page_title="Pneumonia Detection", layout="wide")

def show_single_prediction():
    st.header("Pneumonia Detection from Chest X-rays")
    st.write("Upload a chest X-ray image to detect if it shows signs of pneumonia.")

    # --- Sample images for download ---
    st.markdown("**Download sample images to try the app:**")
    col1, col2 = st.columns(2)
    with col1:
        with open("src/app/samples/sample_normal.jpg", "rb") as f:
            st.download_button("Download NORMAL Sample", f, file_name="sample_normal.jpg", mime="image/jpeg")
    with col2:
        with open("src/app/samples/sample_pneumonia.jpg", "rb") as f:
            st.download_button("Download PNEUMONIA Sample", f, file_name="sample_pneumonia.jpg", mime="image/jpeg")
    st.write(":arrow_down: Download and drag one of the above images into the uploader below.")
    # --- End sample images ---
    
    @st.cache_resource
    def load_pneumonia_model():
        model = PneumoniaCNN()
        model.load_state_dict(torch.load('models/pneumonia_model.pth', map_location=torch.device('cpu')))
        model.eval()
        return model
    
    try:
        model = load_pneumonia_model()
    except:
        st.error("Model file not found. Please train the model first.")
        return
    
    uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-ray", use_column_width=True)
        temp_path = "temp_image.jpg"
        image.save(temp_path)
        if st.button("Detect Pneumonia"):
            with st.spinner("Analyzing the X-ray..."):
                prediction = predict_image(model, temp_path, device='cpu')
                st.write("---")
                st.subheader("Results")
                prob_pneumonia = prediction * 100
                st.progress(prob_pneumonia / 100)
                st.write(f"Probability of Pneumonia: {prob_pneumonia:.2f}%")
                if prob_pneumonia > 50:
                    st.error("⚠️ Pneumonia Detected")
                    st.write("The model has detected signs of pneumonia in the X-ray. Please consult a healthcare professional for proper diagnosis and treatment.")
                else:
                    st.success("✅ No Pneumonia Detected")
                    st.write("The model did not detect signs of pneumonia in the X-ray. However, please consult a healthcare professional for proper medical advice.")
            os.remove(temp_path)

def show_evaluation():
    st.header("Model Evaluation Results")
    # Show test accuracy and classification report if available
    if os.path.exists("evaluation_report.txt"):
        with open("evaluation_report.txt", "r") as f:
            report = f.read()
        st.text(report)
    else:
        st.info("Run the evaluation script to generate evaluation_report.txt.")
    # Show confusion matrix image
    if os.path.exists("confusion_matrix.png"):
        st.subheader("Confusion Matrix")
        st.image("confusion_matrix.png", caption="Confusion Matrix", use_column_width=False)
    else:
        st.info("Run the evaluation script to generate confusion_matrix.png.")
    # Show test predictions image
    if os.path.exists("test_predictions.png"):
        st.subheader("Sample Test Predictions")
        st.image("test_predictions.png", caption="Test Predictions", use_column_width=True)
    else:
        st.info("Run the evaluation script to generate test_predictions.png.")

def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Go to", ["Single Image Prediction", "Model Evaluation"])
    if app_mode == "Single Image Prediction":
        show_single_prediction()
    elif app_mode == "Model Evaluation":
        show_evaluation()

if __name__ == "__main__":
    main() 