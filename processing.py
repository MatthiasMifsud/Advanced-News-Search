from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

class Text_Processor:
    def __init__(self):
        self.stopword_set = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def preprocessing(self, text):
        cleaned_words = word_tokenize(re.sub(r'\s+', ' ', re.sub(r'[^A-Za-z0-9 ]+', '', text)).lower().strip())
        filtered_words = [self.lemmatizer.lemmatize(word) for word in cleaned_words if word not in self.stopword_set]
        filtered_text = ' '.join(filtered_words)
        return filtered_text
    