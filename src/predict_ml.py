import sys
sys.path.append("../src")

from preprocess import preprocess_string

def predict_string_with_ml(model, vectorizer, text):
    decoder = {
        0:'negative',
        1:'positive'
    }
    
    cleaned = preprocess_string(text)
    X_input = vectorizer.transform([cleaned])
    prediction = model.predict(X_input)
    
    return decoder[prediction[0]]