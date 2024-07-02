import streamlit as st
import joblib
import spacy

# SpaCy modelini yükleyin
nlp = spacy.load('en_core_web_sm')

# En iyi modeli yükleme
best_model = joblib.load('best_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Metin ön işleme fonksiyonu
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

# Streamlit uygulaması
st.title("Text Classification App")
st.write("With this application, you can classify the text you enter according to physics, chemistry and biology categories.")

# Metin girişi alanı
user_input = st.text_area("Enter text", "")

if st.button("Enter text"):
    if user_input.strip() == "":
        st.warning("Please enter text.")
    else:
        # Metin ön işleme
        processed_input = preprocess_text(user_input)
        # TF-IDF vektörleştirme
        vectorized_input = vectorizer.transform([processed_input])
        # Modeli kullanarak sınıflandırma
        prediction = best_model.predict(vectorized_input)[0]
        # Sonucu gösterme
        st.success(f"This text is associated with {prediction}.")