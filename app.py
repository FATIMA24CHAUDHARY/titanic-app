import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

model = pickle.load(open("svc_trained_model.pkl", "rb"))

# Same category lists as the notebook (Step 5.1)
pclass_opts = ["First", "Second", "Third"]
gender_opts = ["Male", "Female"]
sibling_opts = ["Zero", "One", "Two", "Three"]
embarked_opts = ["Southampton", "Cherbourg", "Queenstown"]

def make_encoder(values):
    enc = LabelEncoder()
    enc.fit(values)
    return enc

pclass_enc = make_encoder(pclass_opts)
gender_enc = make_encoder(gender_opts)
sibling_enc = make_encoder(sibling_opts)
embarked_enc = make_encoder(embarked_opts)

st.title("Titanic Passenger Survival Prediction")

pclass = st.selectbox("Passenger class", pclass_opts)
gender = st.selectbox("Gender", gender_opts)
sibling = st.selectbox("Siblings aboard", sibling_opts)
embarked = st.selectbox("Port embarked", embarked_opts)

if st.button("Predict"):
    X = pd.DataFrame({
        "PClass": [pclass_enc.transform([pclass])[0]],
        "Gender": [gender_enc.transform([gender])[0]],
        "Sibling": [sibling_enc.transform([sibling])[0]],
        "Embarked": [embarked_enc.transform([embarked])[0]],
    })
    if model.predict(X)[0] == 1:
        st.success("Prediction: SURVIVED")
    else:
        st.error("Prediction: NOT SURVIVED")