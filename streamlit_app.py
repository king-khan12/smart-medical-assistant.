
import streamlit as st
from transformers import pipeline

# Title and user input fields
st.title("Smart Medical Assistant - Symptom Collection")

name = st.text_input("What is your name?")
age = st.number_input("Enter your age:", min_value=0, max_value=120)
gender = st.selectbox("Select your gender:", ["Male", "Female", "Other"])

st.subheader("Please answer the following symptom-related questions:")

chest_pain = st.radio("Do you have chest pain?", ("Yes", "No"))
breathing_difficulty = st.radio("Do you have difficulty breathing?", ("Yes", "No"))
fever = st.radio("Do you have a fever?", ("Yes", "No"))
rash = st.radio("Do you have a skin rash?", ("Yes", "No"))
headache = st.radio("Are you experiencing headaches?", ("Yes", "No"))

pain_severity = st.slider("If you have pain, how severe is it? (0 = None, 10 = Worst possible)", 0, 10)

# Specialist prediction using a pretrained DistilBERT model
classifier = pipeline("text-classification", model="./smart_medical_model")

def get_specialist(symptom_text):
    prediction = classifier(symptom_text)
    return prediction[0]['label']

if st.button("Submit"):
    collected_data = {
        "name": name,
        "age": age,
        "gender": gender,
        "symptoms": {
            "chest_pain": chest_pain,
            "breathing_difficulty": breathing_difficulty,
            "fever": fever,
            "rash": rash,
            "headache": headache,
            "pain_severity": pain_severity
        }
    }
    # Prepare symptom text to classify
    symptom_text = f"{'Chest pain, ' if chest_pain == 'Yes' else ''}" +                    f"{'Breathing difficulty, ' if breathing_difficulty == 'Yes' else ''}" +                    f"{'Fever, ' if fever == 'Yes' else ''}" +                    f"{'Rash, ' if rash == 'Yes' else ''}" +                    f"{'Headache' if headache == 'Yes' else ''}"
    
    # Get the specialist prediction
    specialist = get_specialist(symptom_text)
    st.write(f"Recommended Specialist: {specialist}")

    st.json(collected_data)
