'''
This module contains functions for cleaning/preprocessing text
'''
# Import packages/libraries
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.util import ngrams
from nltk.corpus import stopwords, words
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import gensim
from gensim.models import Phrases
import pickle

# Function for converting British English to American English
def am_eng(text, ab_dict):
    '''
    Takes in a string of text and replaces British English words with American English equivalent
    '''
    for gb, us in ab_dict.items():
        text = text.replace(gb, us)
    return text

# Function for spell checking
from spellchecker import SpellChecker

def spell_check(text):
    '''
    Takes in a list of word-tokenized text, spell checks each word, creates new list containining corrected words,
    and returns list of corrected words
    '''
    spell = SpellChecker()
    misspelled = spell.unknown(text)
    corrected = []
    for word in text:
        if word not in misspelled:
            corrected.append(word)
        else:
            corrected_word = spell.correction(word)
            corrected.append(corrected_word)
    return corrected

# Function to remove spelled out numbers
def remove_spelled_out_numbers(text):
    '''
    Takes in a string of text, word-tokenizes text, converts spelled out numbers to integer equivalent,
    removes integer, and joins tokenized text into string
    '''
    from word2number import w2n

    tokens = word_tokenize(text)
    numbers_removed = []
    
    for token in tokens:
        try:
            if type(w2n.word_to_num(token)) == 'int':
                continue
        except:
            numbers_removed.append(token)
    
    return ' '.join(numbers_removed)

# Function for filtering POS & lemmatizing tokenized texts
# Code adapted from https://medium.com/@gaurav5430/using-nltk-for-lemmatizing-sentences-c1bfff963258
def filter_pos_lemmatize(text):
    '''
    Takes in a list of word-tokenized text, tags each token with POS, filters out only nouns,
    lemmatizes nouns, and returns list of lemmatized nouns
    '''
    # Instantiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Function to convert nltk tags to wordnet tags
    def nltk_tag_to_wordnet_tag(nltk_tag):
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:          
            return None

    # Find POS tag for each token
    nltk_tagged = nltk.pos_tag(text)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)

    # Lemmatize, filter for nouns & adjectives
    lemmatized_words = []
    for word, tag in wordnet_tagged:
        if tag == 'n':
            lemmatized_words.append(lemmatizer.lemmatize(word, tag))
    
    return lemmatized_words

# Function/pipeline for preprocessing text
def preprocess(text):
    '''
    Takes in text, pre-processes it (cleans, converts British English words to American English words,
    tokenizes, lemmatizes, removes short words, removes non-noun words), and joins tokens together into documents
    '''
    # Remove content within parentheses
    clean_text = re.sub(r'\([^()]*\)', ' ', text)
    
    # Remove punctuation EXCEPT FOR '-'
    punctuation = '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'
    clean_text = re.sub('[%s]' % re.escape(punctuation), '', clean_text)
    
    # Replace '-' with ' '
    clean_text = clean_text.replace('-', ' ')
    
    # Remove numbers
    clean_text = re.sub('\w*\d\w*', ' ', clean_text)
    
    # Convert to lowercase
    clean_text = clean_text.lower()
    
    # Remove spelled out numbers
    clean_text = remove_spelled_out_numbers(clean_text)
    
    # Convert British English to American English
    clean_text = am_eng(clean_text, ab_dict)
    
    # Remove basic stop words
    filtered_words = [word for word in word_tokenize(clean_text) if word not in stopwords_l]
    clean_text = ' '.join(filtered_words)
    
    # Lemmatize: code adapted from https://medium.com/@gaurav5430/using-nltk-for-lemmatizing-sentences-c1bfff963258
    lemmatized_words = filter_pos_lemmatize(clean_text)
    
    # Remove words <4 letters in length
    long_words = [word for word in lemmatized_words if len(word) >= 4]
    
    # Join all of the tokens into single string
    doc = ' '.join(long_words)
    
    return doc

# Pipeline for cleaning & preprocessing text
# Function to clean text
def clean(text):
    '''
    Takes in string of text, cleans text, word-tokenizes text, removes stop words, and returns list
    cleaned tokens
    '''
    # Remove punctuation EXCEPT FOR '-'
    punctuation = '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'
    clean_text = re.sub('[%s]' % re.escape(punctuation), '', text)
    
    # Replace '-' with ' '
    clean_text = clean_text.replace('-', ' ')
    
    # Remove numbers
    clean_text = re.sub('\w*\d\w*', '', clean_text)
    
    # Convert to lowercase
    clean_text = clean_text.lower()
    
    # Remove spelled out numbers
    clean_text = remove_spelled_out_numbers(clean_text)
    
    # Import British English to American English dictionary
    infile = open('ab_dict.pkl','rb')
    ab_dict = pickle.load(infile)
    infile.close()
    
    # Convert British English to American English
    clean_text = am_eng(clean_text, ab_dict)
    
    # Stopwords
    stop_words = list(set(stopwords.words('english')))
    custom_stopwords = ["covid-19", "sars-cov-2", "coronavirus", "sars", "COVID-19", "SARS-CoV-2", "SARS", "CoV-2",
                        "covid", "COVID", "CoV", "cov", "corona", "coronaviruses", "n_cov", "ncov", "covs", "severe acute respiratory syndrome coronavirus 2",
                        "background", "objective", "objectives", "methods", "results", "conclusion", "conclusions"]
    stopwords_l = stop_words + custom_stopwords
    
    # Remove stop words
    filtered_words = [word for word in word_tokenize(clean_text) if word not in stopwords_l]
    
    return filtered_words

# Function to make bigrams & trigrams
def make_trigrams(texts, threshold):
    '''
    Builds bigram & trigram models using gensim Phrases and 
    '''
    bigram = gensim.models.Phrases(texts, min_count=1, threshold=threshold)
    trigram = gensim.models.Phrases(bigram[texts], threshold=threshold)

    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

# Function to further process text
def process(texts):
    '''
    Takes in list of tokenized text; clubs bigrams & trigrams; tags POS, filters for nouns, and
    lemmatizes; filters out words <4 letters long, and joins words together into document
    '''
    # Make bigrams/trigrams
    grammed_text = make_trigrams(texts, 500)
    
    # Lemmatize and filter for nouns and adjectives
    lemmatized_text = [filter_pos_lemmatizexs(doc) for doc in grammed_text]
    
    # Remove words <4 letters in length
    long_words_text = []
    for doc in lemmatized_text:
        lemmatized_words = [word for word in doc if len(word) >=4]
        long_words_text.append(lemmatized_words)
    
    # Join words
    texts = [' '.join(doc) for doc in long_words_text]
    
    return texts

# Pipeline
def pp_pipeline(docs_list):
    '''
    Takes in list of texts/documents, calls text cleaning and processing functions,
    and returns list of cleaned and pre-processed (word-tokenized) text
    '''
    cleaned_docs = [clean(doc) for doc in docs_list]
    return process(cleaned_docs)