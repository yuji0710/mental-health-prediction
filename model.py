# ==============================
# INSTALL
# ==============================
# !pip install sentence-transformers scikit-learn pandas -q

# ==============================
# IMPORTS
# ==============================
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv("/kaggle/input/datasets/harshmodi0710/mental-health-2/demo.csv")

# ==============================
# BALANCE DATASET (SAFE)
# ==============================
samples_per_class = 1500

df = df.groupby('status').apply(
    lambda x: x.sample(min(len(x), samples_per_class), random_state=42)
).reset_index(drop=True)

print("📊 Balanced Data:")
print(df['status'].value_counts())

# ==============================
# PREPARE DATA
# ==============================
texts = df['statement'].astype(str)
labels = df['status']

# ==============================
# TF-IDF FEATURES
# ==============================
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(texts)

# ==============================
# SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)

# ==============================
# TRAIN ML MODEL (IMPROVED)
# ==============================
clf = LogisticRegression(max_iter=2000)
clf.fit(X_train, y_train)

# ==============================
# EVALUATE
# ==============================
y_pred = clf.predict(X_test)
print("\n🔥 TF-IDF Accuracy:", accuracy_score(y_test, y_pred))
print("=" * 60)

# ==============================
# LOAD SENTENCE TRANSFORMER
# ==============================
model = SentenceTransformer('all-MiniLM-L6-v2')

# ==============================
# STRONG REFERENCE SET
# ==============================
ref_texts = [

# depression
"I feel empty inside",
"I feel hopeless and tired",
"I feel like giving up",
"I feel lost in life",
"I feel nothing anymore",

# anxiety
"I feel nervous all the time",
"My mind is racing",
"I feel stressed and overwhelmed",
"I cannot relax",
"I am constantly worried",

# fear
"I am scared",
"I feel afraid",
"I am terrified",
"I feel unsafe",

# anger
"I am very angry",
"I feel frustrated",
"I am irritated",
"I feel rage",

# sadness
"I feel sad",
"I feel low",
"I feel unhappy",

# normal
"I went to college",
"I had lunch",
"I did my work",
"I followed my routine",
"I had a normal day",

# happy
"I feel happy",
"I am enjoying life",
"I feel great",
"I am excited",
"I feel good"
]

ref_labels = [
"depression","depression","depression","depression","depression",
"anxiety","anxiety","anxiety","anxiety","anxiety",
"fear","fear","fear","fear",
"anger","anger","anger","anger",
"sadness","sadness","sadness",
"normal","normal","normal","normal","normal",
"happy","happy","happy","happy","happy"
]

# Encode once
ref_emb = model.encode(ref_texts, convert_to_tensor=True)

# ==============================
# PREDICTION FUNCTIONS
# ==============================
def predict_tfidf(text):
    vec = tfidf.transform([text])
    return clf.predict(vec)[0]

def predict_sbert(text):
    emb = model.encode([text], convert_to_tensor=True)
    scores = util.cos_sim(emb, ref_emb)
    idx = torch.argmax(scores).item()
    return ref_labels[idx]

# ==============================
# FINAL HYBRID DECISION
# ==============================
def final_predict(text):
    tfidf_pred = predict_tfidf(text)
    sbert_pred = predict_sbert(text)

    # If TF-IDF says normal but SBERT detects something else → trust SBERT
    if tfidf_pred.lower() == "normal" and sbert_pred != "normal":
        return sbert_pred

    # If both same → return
    if tfidf_pred.lower() == sbert_pred:
        return tfidf_pred

    # Otherwise → prefer SBERT
    return sbert_pred

# ==============================
# TEST CASES
# ==============================
tests = [
    "I smile but cry inside",
    "I feel very happy today",
    "I am nervous all the time",
    "I am angry right now",
    "I feel scared",
    "I went to college and came back",
    "I feel empty and tired",
    "My mind never stops thinking",
    "I am okay but something feels wrong"
]

print("\n🧪 FINAL RESULTS")
print("=" * 60)

for t in tests:
    print(f"Text: {t}")
    print(f"TF-IDF: {predict_tfidf(t)}")
    print(f"Sentence: {predict_sbert(t)}")
    print(f"Final: {final_predict(t)}")
    print("-" * 60)

# ==============================
# CHAT MODE
# ==============================
print("\n💬 CHAT MODE (type 'exit' to stop)\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    print("👉 Final Prediction:", final_predict(user_input))
    print("-" * 50)



    # ==============================
# SAVE MODEL AS PKL
# ==============================
import pickle

save_data = {
    "tfidf": tfidf,
    "clf": clf,
    "sbert_model_name": "all-MiniLM-L6-v2",
    "ref_texts": ref_texts,
    "ref_labels": ref_labels,
    "ref_emb": ref_emb.cpu().numpy(),  # save as numpy
}

with open("mental_health_model.pkl", "wb") as f:
    pickle.dump(save_data, f)

print("✅ Model saved successfully as mental_health_model.pkl")