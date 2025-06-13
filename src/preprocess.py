import re
from natasha import Segmenter, NewsEmbedding, NewsMorphTagger, MorphVocab, Doc
from nltk.corpus import stopwords

segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
morph_vocab = MorphVocab()

stop = set(stopwords.words('russian')) - {'не', 'ни'}

def preprocess_string(string):
    if not isinstance(string, str) or len(string.strip()) == 0: # fool-protection
        return ''
    
    string = clean_string(string)
    string = lemmantize_words(string)
    string = filter_stop_words(string)
    
    return string


def filter_stop_words(text):
    words = []
    for word in text.split():
        if word in stop or len(word) < 2:
            continue
        words.append(word)
        
    return ' '.join(words)


def lemmantize_words(text):
    lemmas = []
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    
    try:
        for token in doc.tokens:
            token.lemmatize(morph_vocab)
            lemmas.append(token.lemma)
                
    except Exception as e:
        print(f"Ошибка при обработке текста: {text[:30]}... → {e}")
        return ''
    
    return ' '.join(lemmas)


def clean_string(text):
    text = re.sub(r'<.*?>', ' ', text) # removing html-tags from text
    text = re.sub(r'http\S+|\S+@\S+', ' ', text) # removing links
    text = re.sub(r'[^а-яё\s]', ' ', text.lower()) # remove all non-letter symbols
    cleaned_text = re.sub(r'\s+', ' ', text).strip() # remove double or more spaces
    
    return cleaned_text