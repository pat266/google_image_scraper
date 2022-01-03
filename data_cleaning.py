from bs4 import BeautifulSoup
import re
import unicodedata
import nltk
nltk.download('punkt')
from nltk.tokenize.toktok import ToktokTokenizer
import spacy

# Got the contraction map from arturomp
# (https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python)
CONTRACTION_MAP = {
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I would",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}

class DataCleaner:
    def __init__(self):
        self.nlp = None
        self.tokenizer = None
        self.stopword_list = None

    # Clean sentence, leaving only the keywords behind
    def preprocess_split_corpus(self, text):
        # Further cleaning and separating the sentence into a list of words
        words = self.clean_data([text])[0]
        return ' '.join([words]).split()
        # return [doc.split(" ") for doc in text]

    def clean_data(self, data, remove_digits=False):
        if self.tokenizer is None:
            self.nlp = spacy.load('en_core_web_md', parse=True, tag=True, entity=True)
            self.tokenizer = ToktokTokenizer()
        if self.stopword_list is None:
            # Initialize stop word list; remove 'no' and 'not' from the list to keep meaning in the sentence
            self.stopword_list = nltk.corpus.stopwords.words('english')
            self.stopword_list.remove('no')
            self.stopword_list.remove('not')
        
        return self.normalize_corpus(data, remove_digits)

    # Preprocess data/input
    def preprocess_input(self, doc, remove_digits=False):
        # Remove accented characters
        doc = self.remove_accented_chars(doc)

        # Expand contractions
        doc = self.expand_contractions(doc)

        # Lowercase the text
        doc = doc.lower()

        # Remove extra newlines
        doc = re.sub(r'[\r|\n|\r\n]+', ' ', doc)

        # Lemmatize text
        doc = self.lemmatize_text(doc)

        # Remove special characters and\or digits
        # insert spaces between special characters to isolate them
        special_char_pattern = re.compile(r'([{.(-)!}])')
        doc = special_char_pattern.sub(" \\1 ", doc)
        doc = self.remove_special_characters(doc, remove_digits=remove_digits)

        # Remove extra whitespaces
        doc = re.sub(' +', ' ', doc)

        # Remove stopwords
        doc = self.remove_stopwords(doc)

        return doc

    # Normalize our document with the preprocessing functions
    def normalize_corpus(self, corpus, remove_digits=False):
        normalized_corpus = []
        # Normalize each document in the corpus
        for doc in corpus:
            if doc is not None:
                doc = self.preprocess_input(doc, remove_digits)
            normalized_corpus.append(doc)

        return normalized_corpus

    # Remove accented characters
    def remove_accented_chars(self, text):
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return text

    # Expand contractions
    def expand_contractions(self, text):
        contractions_pattern = re.compile('({})'.format('|'.join(CONTRACTION_MAP.keys())),
                                          flags=re.IGNORECASE|re.DOTALL)
        def expand_match(contraction):
            match = contraction.group(0)
            first_char = match[0]
            expanded_contraction = CONTRACTION_MAP.get(match)\
                if CONTRACTION_MAP.get(match)\
                else CONTRACTION_MAP.get(match.lower())
            expanded_contraction = first_char+expanded_contraction[1:]
            return expanded_contraction

        expanded_text = contractions_pattern.sub(expand_match, text)
        expanded_text = re.sub("'", "", expanded_text)
        return expanded_text

    # Remove special characters
    def remove_special_characters(self, text, remove_digits=False):
        pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
        text = re.sub(pattern, '', text)
        return text

    # Lemmatize the text
    def lemmatize_text(self, text):
        text = self.nlp(text)
        text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
        return text

    # Remove stop words
    def remove_stopwords(self, text, is_lower_case=True):
        tokens = self.tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]
        if is_lower_case:
            filtered_tokens = [token for token in tokens if token not in self.stopword_list]
        else:
            filtered_tokens = [token for token in tokens if token.lower() not in self.stopword_list]
        filtered_text = ' '.join(filtered_tokens)
        return filtered_text