import streamlit as st
from PIL import Image, UnidentifiedImageError
from model.vit_inference import get_predictions
from app.plant_insight_agent import generate_insight_for_plant



# Streamlit UI
st.title("Medicinal Leaf Classifier")
st.write("Upload a leaf image to identify the medicinal plant and get insights.")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Predict top-3 labels
        top_labels, top_scores, label_list = get_predictions(image)

        st.markdown("### 🔍 Top Predictions")
        for i in range(3):
            st.write(f"**{i+1}. {top_labels[i]}** — Confidence: {top_scores[i]:.2%}")

        st.markdown("### ✅ Is the top prediction correct?")
        user_feedback = st.radio("Your answer:", ["Yes", "No"], horizontal=True)

        if user_feedback == "Yes":
            final_label = top_labels[0]
        else:
            final_label = st.selectbox("Select the correct plant name:", label_list)

        if st.button("Generate Insights"):
            with st.spinner(f"Fetching insights for **{final_label}** using Gemini..."):
                insights = generate_insight_for_plant(final_label)
                st.markdown("### 🌿 Plant Insights")

                for section, content in insights.items():
                    st.markdown(f"#### {section}")
                    if content:
                        for line in content:
                            st.markdown(f"{line}")
                    else:
                        st.markdown("_No data available._")

    except UnidentifiedImageError:
        st.error("Could not read image. Please upload a valid image file.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
