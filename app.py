import streamlit as st
import pickle
import torch
from sentence_transformers import SentenceTransformer, util

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    with open("mental_health_model.pkl", "rb") as f:
        data = pickle.load(f)

    tfidf = data["tfidf"]
    clf = data["clf"]
    ref_texts = data["ref_texts"]
    ref_labels = data["ref_labels"]
    ref_emb = torch.tensor(data["ref_emb"])

    sbert_model = SentenceTransformer(data["sbert_model_name"])

    return tfidf, clf, ref_texts, ref_labels, ref_emb, sbert_model


tfidf, clf, ref_texts, ref_labels, ref_emb, sbert = load_model()


# ==============================
# PREDICTION FUNCTION
# ==============================
def final_predict(text):
    # TF-IDF prediction
    vec = tfidf.transform([text])
    tfidf_pred = clf.predict(vec)[0]

    # SBERT prediction
    emb = sbert.encode([text], convert_to_tensor=True)
    scores = util.cos_sim(emb, ref_emb)
    idx = torch.argmax(scores).item()
    sbert_pred = ref_labels[idx]

    # Hybrid decision
    if tfidf_pred.lower() == "normal" and sbert_pred != "normal":
        return sbert_pred

    if tfidf_pred.lower() == sbert_pred:
        return tfidf_pred

    return sbert_pred


# ==============================
# STREAMLIT UI
# ==============================
st.set_page_config(page_title="Mental Health Detector", page_icon="🧠", layout="centered")

st.title("🧠 Mental Health Detection System")
st.write("Enter any text and the model will detect the emotional/mental state.")

text = st.text_area("📝 Enter your text below:")

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        result = final_predict(text)
        st.success(f"💡 Prediction: **{result.upper()}**")


st.markdown("---")
st.caption("Model: Hybrid TF-IDF + SBERT | Built by Harsh Modi")