import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from sklearn.metrics import pairwise_distances

def main():
    
    st.title('COVID-19 Research Abstract Recommender')
    
    ## Load data and model
    # Topics
    topic_names = {0: 'clinical recommendations',
                1: 'pathophysiology',
                2: 'prognosis',
                3: 'children',
                4: 'cancer',
                5: 'mental health',
                6: 'diagnosis/serology',
                7: 'resources & technology',
                8: 'vaccine',
                9: 'predictions/forecasting',
                10: 'transmission',
                11: 'global impact',
                12: 'pregnancy',
                13: 'surgery',
                14: 'treatment',
                15: 'microbiology',
                16: 'education',
                17: 'clinical features',
                18: 'deaths',
                19: 'healthcare workers & PPE'}
    topic_labels = list(topic_names.values())
    
    doc_word = pickle.load(open('doc_word.pkl', 'rb'))
    doc_topic = pickle.load(open('doc_topic.pkl', 'rb'))
    tfidf = pickle.load(open('tfidf.pkl', 'rb'))
    pmids = pickle.load(open('pmids.pkl', 'rb'))
    
    # Dataframes
    doc_word_df = pd.DataFrame(doc_word.toarray(), index=pmids, columns=tfidf.get_feature_names())
    doc_topic_df = pd.DataFrame(doc_topic.round(5), index=pmids, columns=topic_labels)
    
    # Calculate cosine distances
    dists = pairwise_distances(doc_topic_df, metric='cosine')
    dists_df = pd.DataFrame(data=dists, index=doc_word_df.index, columns=doc_word_df.index)

    ## User input
    # Look at abstracts/articles on PubMed
    link = '[Explore COVID-19 research on PubMed](https://pubmed.ncbi.nlm.nih.gov/?term=covid-19)'
    st.markdown(link, unsafe_allow_html=True)
    
    # Enter PMID for abstract of interest
    chosen_abstract = st.text_input('Enter the PubMed ID (PMID) of an abstract/paper')
    
    # Return similar abstract/article
    def similar_abstracts(chosen_abstract):
        abstracts_summed = dists[chosen_abstract].sum(axis=1)
        abstracts_summed = abstracts_summed.sort_values(ascending=True)
        mask = ~abstracts_summed.index.isin(chosen_abstract)
        ranked_abstracts = abstracts_summed.index[mask]
        ranked_abstracts = ranked_abstracts.tolist()
        recommendations = ranked_abstracts[:5]
        return recommendations
    
    recs = similar_abstracts(chosen_abstract)
    for i in range(1,len(recs)+1):
        base_url = 'https://pubmed.ncbi.nlm.nih.gov/{}/'
        rec_url = base_url.format(recommendations[i])
    
    # if st.button('Abstract #1'):
    #     components.iframe(rec_1_url, height=600, scrolling=True)
    # elif st.button('Abstract #2'):
    #     components.iframe(rec_2_url, height=600, scrolling=True)
    # elif st.button('Abstract #3'):
    #     components.iframe(rec_3_url, height=600, scrolling=True)
    # elif st.button('Abstract #4'):
    #     components.iframe(rec_4_url, height=600, scrolling=True)
    # elif st.button('Abstract #5'):
    #     components.iframe(rec_5_url, height=600, scrolling=True)
    
if __name__=='__main__': 
    main()