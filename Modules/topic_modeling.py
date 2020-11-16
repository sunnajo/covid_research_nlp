'''
This module contains functions for topic modeling
'''
import pandas as pd
import numpy as np
from random import randint
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import gensim
from gensim import matutils, models
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from gensim.models.nmf import Nmf
import scipy.sparse
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD, NMF, PCA
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

# Function for replacing certain terms
def replace_words(text_l):
    '''
    Takes in list of preprocessed abstract texts, replaces erroneous terms with correct terms, and returns list of abstract texts
    '''
    text_l = [text.replace("york_city", "new_york_city") for text in text_l]
    text_l = [text.replace("randomisation", "randomization") for text in text_l]
    return text_l

# Function to remove stopwords from text
def remove_stopwords(text):
    '''
    Takes in string of text, tokenizes text, removes stopwords, and returns edited string of text
    '''
    custom_stopwords = ["covid-19", "sars-cov-2", "coronavirus", "sars", "COVID-19", "SARS-CoV-2", "SARS", "CoV-2", "covid", "COVID", "CoV", "cov", "corona", "coronaviruses", "n_cov", "covs", "severe acute respiratory syndrome coronavirus 2",
                    "introduction", "introductions", "background", "objective", "objectives", "methods", "results", "conclusion", "conclusions", "study", "research", "paper", "medline", "pubmed", "cochrane", "arxiv",
                   "patient", "patients", "virus", "viruses", "none", "n_cov", "medicine", "time", "seconds", "hour", "hours", "day", "days", "year", "years",
                    "week", "weeks", "month", "months", "date", "dates", "anti", "mean", "means", "times", "some", "many", "multiple", "few",
                    "several", "person", "people", "individual", "individuals", "case", "disease", "illness", "syndrome", "sick", "sickness",
                    "human", "situation", "treat", "treatment", "therapy", "drug", "cause", "issue", "line", "number", "way", "method",
                    "ncov", "covs", "health", "care", "infection", "symptom", "symptoms", "type", "data", "group", "level",
                   "test", "event", "incident", "result", "center", "effect", "feature", "pandemic", "characteristic",
                   "practice", "evidence", "acute", "finding", "role", "service", "procedure", "measure", "healthcare", "health", "care",
                   "factor", "department", "period", "region", "area", "rate", "system", "sample", "value", "review", "unit",
                   "report", "analysis", "result", "manifestation", "review", "impact", "hcovs", "article", "involvement", "need", "material",
                   "sign", "presentation", "participant", "unit", "year_old", "year_old_man", "year_old_woman", "york", "camel", "post",
                   "difference", "similarity", "novel", "outbreak", "animal", "knowledge", "epidemic", "complication", "literature", "strain",
                    "fatality", "pathogenesis", "pathogen", "protein", "increase", "value", "performance", "blood", "province", "city",
                    "china", "wuhan", "hubei", "hubei_province", "middle_east", "mers", "term", "immune", "york",
                   "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
    add_stopwords = custom_stopwords
    add_stopwords = list(set(add_stopwords))
    
    words = [word for word in word_tokenize(text) if word not in add_stopwords]
    text = ' '.join(words)
    return text

# Pipeline for additional preprocessing of text
def prep_text(text_l):
    '''
    Takes in list of preprocessed abstract texts, calls functions to replace erroneous terms and remove additional stopwords, and returns
    list of further preprocessed texts
    '''
    text_l = replace_words(text_l)
    corpus = [remove_stopwords(text) for text in text_l]
    return corpus

# Function for displaying topics
def display_topics(model, feature_names, num_top_words, topic_names=None):
    '''
    Takes in model, feature names, the number of top words to return, and topic names, and returns n keywords for for each topic for given model
    '''
    for idx, topic in enumerate(model.components_):
        if not topic_names or not topic_names[idx]:
            print("\nTopic ", idx)
        else:
            print("\nTopic: '", topic_names[idx],"'")
        print(", ".join([feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]]))

# Function for testing different numbers of topics
def compare_topics(vectorizer, text, n_components):
    '''
    Takes in vectorizer, string of text, and number of components/topics and returns topics with 15 keywords for each topic
    '''
    # Vectorize
    doc_word = vectorizer.fit_transform(text)
    
    # Instantiate and fit NMF model
    nmf_model = NMF(n_components=n_components, init='nndsvd', random_state=0)
    doc_topic = nmf_model.fit_transform(doc_word)
    
    # Display topics
    print('{} topics: \n'.format(n_components))
    display_topics(nmf_model, vectorizer.get_feature_names(), 15)

# Function for retrieving top n words for each topic
def topic_top_words(vectorizer, num_topics, model, topic_labels):
    '''
    Takes in vectorizer, number of topics, and model and returns top 15 words for each topic
    '''
    feat_names = vectorizer.get_feature_names()
    word_dict = {};
    for i in range(num_topics):
        # For each topic, obtain the largest values, and add the words they map to into the dictionary
        words_ids = model.components_[i].argsort()[:-15 - 1:-1]
        words = [feat_names[key] for key in words_ids]
        word_dict[topic_labels[i]] = words
    return pd.DataFrame(word_dict)

# Pipeline for vectorizing and creating document-term matrix
def vectorize_dtm(vectorizer_method, text, min_df, max_df):
    '''
    Takes in vectorizer, list of texts, and min_df & max_df parameters,
    vectorizes into corpus, and creates and returns document-term matrix
    '''
    if vectorizer_method == 'count':
        vectorizer = CountVectorizer(strip_accents='unicode',
                                     max_df=max_df, min_df=min_df)
    elif vectorizer_method == 'tfidf':
        vectorizer = TfidfVectorizer(strip_accents='unicode',
                                     max_df=max_df, min_df=min_df)
    
    # Create document-term matrix
    dtm = vectorizer.fit_transform(text)
    
    return vectorizer, dtm

# Pipeline for gensim LDA model
def gensim_lda(text_l, num_topics):
    '''
    Takes in list of texts and number of topics, tokenizes each text and creates dictinary, creates corpus using
    bag-of-words method, builds gensim LDA model, and displays topics from LDA model 
    '''
    # Create dictionary
    text_tokenized = [word_tokenize(text) for text in text_l]
    id2word = corpora.Dictionary(text_tokenized)

    # Term-document frequency - bag of words
    corpus = [id2word.doc2bow(text) for text in text_tokenized]

    # Build model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=num_topics,
                                                passes=10,
                                                update_every=1,
                                                chunksize=100,
                                                alpha='auto',
                                                per_word_topics=True,
                                                random_state=0)
    
    return lda_model, corpus, id2word

# Pipeline for sklearn LDA model
def sklearn_lda(dtm, n_components):
    '''
    Takes in document-term matrix and number of components, builds sklearn LDA model,
    and returns document-topic matrix
    '''
    # Build model
    lda = LatentDirichletAllocation(n_components=n_components,
                                    random_state=0,
                                    learning_method='online',
                                    verbose=0,
                                    n_jobs=-1)
    lda_topic_matrix = lda.fit_transform(dtm)
    return lda, lda_topic_matrix

# Function to display sklearn LDA model topics
# (code adapted from https://medium.com/analytics-vidhya/topic-modelling-using-latent-dirichlet-allocation-in-scikit-learn-7daf770406c4)
def sklearn_lda_topics(lda_model, vectorizer):
    '''
    Takes in LDA model and vectorizer and returns top 15 words for each topic
    '''
    for index, topic in enumerate(lda_model.components_):
        print(f'Top 15 words for Topic #{index}')
        print([vectorizer.get_feature_names()[i] for i in topic.argsort()[-15:]])
        print('\n')

# Function for computing perplexity and coherence score for LDAmodel
def lda_score(text_l, corpus, dictionary, model):
    '''
    Takes in list of texts, corpus, dictionary, and LDA model, and returns perplexity and c_v coherence score for model
    '''
    # Tokenize texts
    text_tokenized = [word_tokenize(text) for text in text_l]
    
    # Compute perplexity
    print('\nPerplexity: ', model.log_perplexity(corpus))

    # Compute coherence score
    coherence_model_lda = CoherenceModel(model=model, texts=text_tokenized, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_score)

# Function to compute coherence values for different numbers of topics
# (code adapted from https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/#14computemodelperplexityandcoherencescore)
def compute_coherence_values(texts, dictionary, corpus, limit, start=2, step=3):
    '''
    Compute c_v coherence for various numbers of topics

    Parameters:
    ----------
    texts: list of input texts
    dictionary: gensim dictionary
    corpus: gensim corpus
    num_topics: number of topics
    limit: max number of topics

    Returns:
    -------
    model_list: list of LDA topic models
    coherence_values: coherence values corresponding to the LDA model with respective number of topics
    '''
    # Tokenize texts
    text_tokenized = [word_tokenize(text) for text in texts]
    
    coherence_values = []
    model_list = []
    
    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=num_topics,
                                           random_state=0,
                                           passes=10,
                                           update_every=1,
                                           chunksize=100,
                                           alpha='auto',
                                           per_word_topics=True)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=text_tokenized, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

# Function for building and plotting PCA model
def pca_model(vectorizer, dtm, n_components, abstract_labels):
    '''
    Takes in vectorizer, document-term matrix, number of components, and list of abstract labels, builds PCA model,
    obtains components/features and explained variance ratio, and plots explained variance
    '''
    # Convert dtm to dataframe
    dtm_df = pd.DataFrame(dtm.toarray(), index=abstract_labels, columns=vectorizer.get_feature_names())

    # Build PCA model
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(dtm_df)
    
    return pca

def plot_pca(pca):
    '''
    Takes in PCA model and plots explained variance against number of principal components
    '''
    # Plot explained variances
    exp_variances = pca.explained_variance_ratio_.cumsum()
    plt.plot(exp_variances);
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Explained Variance")
    plt.show()

# Pipeline for LSA model
def lsa_model(n_components, dtm, vectorizer, abstract_labels):
    '''
    Takes in number of components, dtm, and vectorizer, builds LSA model, and returns explained variance,
    topic-term matrix as dataframe, and document-topic matrix as dataframe
    '''
    # Build model
    lsa = TruncatedSVD(n_components=n_components, random_state=0)
    doc_topic_matrix = lsa.fit_transform(dtm)
    
    # Explained variance
    print('Explained variance: ', lsa.explained_variance_ratio_)
    
    # Topic-term matrix
    topic_word_df = pd.DataFrame(lsa.components_.round(3),
             index = ['component_{}' for i in range(1,n_components+1)],
             columns = vectorizer.get_feature_names())
    
    # Document-topic matrix
    doc_topic_df = pd.DataFrame(doc_topic_matrix.round(5),
             index = abstract_labels,
             columns = ['component_{}' for i in range(1,n_components+1)])
    
    return lsa, topic_word_df, doc_topic_df

# Pipeline for NMF model
def nmf_model(n_components, dtm, vectorizer, abstract_labels):
    '''
    Takes in number of components, dtm, vectorizer, and list of abstract labels,
    build NMF model, and returns NMF model, document-topic matrix, and topic-term matrix
    '''
    # Build model
    nmf_model = NMF(n_components=n_components, init='nndsvd', random_state=0)

    # Topic labels
    topic_labels = ['topic_{}' for i in range(1,n_components+1)]

    # Create document-topic matrix
    doc_topic_matrix = nmf_model.fit_transform(dtm)
    doc_topic_df = pd.DataFrame(doc_topic_matrix.round(5),
                                index = abstract_labels,
                                columns = topic_labels)
    
    # Create topic-term matrix
    topic_word_df = pd.DataFrame(nmf_model.components_.round(3),
                                 index = topic_labels,
                                 columns = vectorizer.get_feature_names())
    
    return nmf_model, doc_topic_matrix, doc_topic_df, topic_word_df

# Pipeline for final NMF model
def final_nmf_model(text, n_components, abstract_labels, min_df, max_df):
    '''
    Takes in list of texts, number of components, abstract labels, and parameters for vectorization,
    vectorizes text & creates document-term matrix, builds NMF model & creates
    document-topic and topic-term matrices, returns matrices
    '''
    # Vectorize/prepare document-term matrix
    tfidf = TfidfVectorizer(strip_accents='unicode',
                            max_df=max_df, min_df=min_df)
    
    # Document-term matrix
    doc_word = tfidf.fit_transform(text)
    doc_word_df = pd.DataFrame(doc_word.toarray(), index=abstract_labels, columns=tfidf.get_feature_names())
    
    # Build NMF model
    nmf_model = NMF(n_components=n_components, init='nndsvd', random_state=0)

    # Topic labels
    topic_labels = ['topic_{}'.format(i for i in range(n_components+1))]

    # Document-topic matrix
    doc_topic = nmf_model.fit_transform(doc_word)
    doc_topic_df = pd.DataFrame(doc_topic.round(3), index=abstract_labels, columns=topic_labels)
    
    # Topic-term matrix
    topic_word_df = pd.DataFrame(nmf_model.components_.round(3), index=topic_labels, columns=tfidf.get_feature_names())

    return tfidf, nmf_model, doc_word_df, doc_topic_df, topic_word_df

def nmf_coherence_scores(text_l, min_df, max_df):
    '''
    Build Gensim NMF model, calculate coherence scores for various numbers of topics,
    and plot coherence scores against number of topics
    '''
    texts = [word_tokenize(text) for text in text_l]

    # Create a dictionary
    dictionary = corpora.Dictionary(texts)

    # Filter out extremes to limit the number of features
    dictionary.filter_extremes(
        no_below=min_df,
        no_above=max_df
    )

    # Create the bag-of-words format (list of (token_id, token_count))
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Create a list of the topic numbers we want to try
    topic_nums = list(np.arange(5, 75 + 1, 5))

    # Run NMF model and calculate coherence score for each number of topics
    coherence_scores = []

    for num in topic_nums:
        nmf = Nmf(
            corpus=corpus,
            num_topics=num,
            id2word=dictionary,
            chunksize=2000,
            passes=5,
            kappa=.1,
            minimum_probability=0.01,
            w_max_iter=300,
            w_stop_condition=0.0001,
            h_max_iter=100,
            h_stop_condition=0.001,
            eval_every=10,
            normalize=True,
            random_state=42
        )
        
        cm = CoherenceModel(
            model=nmf,
            texts=texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        
        coherence_scores.append(round(cm.get_coherence(), 5))

    # Get the number of topics with the highest coherence score
    scores = list(zip(topic_nums, coherence_scores))
    best_num_topics = sorted(scores, key=operator.itemgetter(1), reverse=True)[0][0]
    print(scores)
    print(best_num_topics)
    
    # Plot coherence scores
    plt.figure(figsize=(8,5))
    plt.plot(topic_nums, coherence_scores, color='r', linewidth=2)
    plt.title('NMF Model Optimization: Coherence Scores', fontsize=16)
    plt.xlabel('Number of topics', fontsize=14)
    plt.ylabel('Coherence score', fontsize=14)
    plt.xticks(np.arange(5,80,5), fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()
    
# Function for choosing number of k-means clusters: elbow method
def kmeans_clusters(max_clusters, dtm):
    '''
    Takes in max number of clusters, determines inertias/SSE for various numbers of clusters,
    and plots inertias/SSEs against number of clusters
    '''
    kmeans_params = {
        "init": "random",
        "n_init": 10,
        "max_iter": 300,
        "random_state": 0
    }

    # List of inertias for different k clusters
    sse = []
    for k in range(1, max_clusters):
        kmeans = KMeans(n_clusters=k, **kmeans_params)
        kmeans.fit(dtm)
        sse.append(kmeans.inertia_)

    # Plot
    plt.plot(range(1, max_clusters), sse)
    plt.xticks(range(1, max_clusters))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.show()
    
# Function for plotting k-means clusters
def kmeans(doc_topic, n_clusters):
    '''
    Takes in document-topic matrix and number of clusters, builds k-means cluster model,
    and plots clusters and centroids
    '''
    # Build k-means clusters model
    km = KMeans(n_clusters=n_clusters, init='random', n_jobs=-1, max_iter=300, random_state=0)
    km.fit(doc_topic)
    
    # Generate random colors
    colors = []
    for i in range(n_clusters):
        colors.append('#%06X' % randint(0, 0xFFFFFF))
    
    # Variables
    X = doc_topic
    y_kmeans = km.fit_predict(X)
    
    # Plot
    plt.figure(figsize=(8,6))
    plt.title('Clusters of Topics')
    for i in range(n_clusters):
        plt.scatter(X[y_kmeans==i,0], X[y_kmeans==i,1], s=50, c=colors[i], label='Cluster {}'.format(i))
    plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], s=100, c='yellow', label = 'Centroids')
    plt.show()