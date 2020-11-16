# Using Natural Language Processing and Unsupervised Learning to Mine COVID-19 Research



## Objectives

- Using natural language processing and topic modeling techniques, determine and analyze themes in COVID-19-related scientific and medical research
- Build a recommender system for similar COVID-19-related abstracts/papers



## Data

- 42,977 abstracts relevant to COVID-19 from January 1-November 5, 2020 obtained through querying [PubMed](https://pubmed.ncbi.nlm.nih.gov/) via Entrez database API
- 2,209,210 terms



## Process & Results

- NLP/text cleaning & preprocessing
  - Used regex, NLTK, and Gensim to clean text (e.g. remove punctuation, numbers (including spelled out numbers), convert all letters to lowercase, convert British English to American English for standardization), tag words with POS
  - Formed bigrams and trigrams as many medical/scientific terms are compound terms
  - Filtered for nouns as these would likely provide the most information
  - Removed words <4 letters in length as these were less likely to be valuable
- Performed exploratory data analysis of data
- Performed exploratory data analysis of preprocessed text to inform my topic modeling process
  - Descriptive statistics: quartiles, median, range, mean
  - Plotted histogram and box plot
  - Looked at most common and least common terms, used this information to set parameters for min_df and max_df for vectorization/creating dictionary for topic modeling
  - Looked for any errors in preprocessing workflow and modified as needed
- Dimensionality reduction/topic modeling
  - Iterations:
    - Tried bag-of-words, count vectorization, and TFIDF vectorization methods
    - Tried various different topic modeling techniques: LDA (sklearn and gensim), PCA, LSA, NMF, Corex
    - Tried k-means clustering
    - Looked at word embeddings with word2vec
  - NMF with TFIDF vectorization yielded the most coherent topics
  - Evaluation: Measured coherence scores for model across different numbers of topics to determine optimal number of topics. 20 topics yielded the highest coherence score (0.56) and this number also appeared to produce the most coherent and well separated topics on eye check.
- Performed exploratory data analysis of topics
- Built recommender system using content-based filtering that returns similar abstracts/papers



## Techniques & Tools

- Natural language processing
  - Regex
  - NLTK
  - Gensim
- Unsupervised learning
  - Dimensionality reduction
    - PCA
    - LSA/SVD
    - NMF
    - LDA
    - Corex
  - Clustering
    - k-means clustering
  - Word embeddings
    - word2vec
- Building recommender systems
- Data visualization
  - matplotlib
  - seaborn
  - plotly



## Results

- EDA of data - *Key Takeaways*

  - The number of abstracts significantly increased over time with the increase being particularly steep from March-July 2020
    - Timeline correlation: WHO declared COVID-19 pandemic in early March, pandemic reached peak in some areas in the summer
  - Top 5 countries in terms of number of abstracts: U.S., China, U.K., Italy, India
    - These countries also rank in top 15 in terms of high quality scientific research output, per 2019 Nature index

- 20 topics:

  1) Clinical recommendations

  2) Pathophysiology

  3) Prognosis

  4) Children

  5) Cancer

  6) Mental health

  7) Diagnosis/Serology 

  8) Resources & Technology

  9) Vaccine

  10) Predictions/Forecasting

  11) Transmission

  12) Global impact

  13) Pregnancy

  14) Surgery

  15) Treatment

  16) Microbiology

  17) Education

  18) Clinical features

  19) Deaths

  20) Healthcare workers & personal protective equipment (PPE)

- EDA of topics - *Key Takeaways*

  - Most common topic: Resources & Technology
  - Topic trends
    - Increasing in prevalence: resources & technology, microbiology, predictions/forecasting, diagnosis/serology, vaccine, education
      - Diagnosis/serology, predictions/forecasting, education, and resources & technology topics are also increasing in relative focus
    - Decreasing in prevalence: global impact, treatment, children, clinical recommendations, surgery, cancer
  - Wide breadth: 50% of topics were represented by >50% of all countries in dataset, all topics were represented by >50% of U.S. states in dataset

  

## Potential Impact

- With the surge in COVID-19 research, it is important to identify which topics are being studied and be aware of trends in research
- Can use recommender system to search for COVID-19-related articles within certain topics/areas of interest



## Future Work

- Correlate analyses with metadata
- Clustering based on study type
- Further topic modeling/clustering: within each topic, month, geographic area