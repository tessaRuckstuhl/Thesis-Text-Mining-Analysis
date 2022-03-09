import collections
import datetime
import logging
import re
import time

import gensim
import nltk
import pandas as pd

nltk.download('wordnet')
ts = time.time()
ts_str = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("myLogger")


# read in data
def read_csv_file(raw_path):
    return pd.read_csv(raw_path, sep=',', engine='python')


# makes bigrams from texts
def make_bigrams(mod, texts):
    return [mod[doc] for doc in texts]


# lemmatizes a word
def get_lemma(word):
    lemma = nltk.wordnet.WordNetLemmatizer()
    return lemma.lemmatize(word)


# clean and tokenize documents
def preprocess_documents(docs, stopwords, synonyms):
    # removal of block code snippets (<code> snippet... </code>)
    documents_cleaned = [re.sub(r'<pre><code.+?</code></pre>', ' ', document,
                                flags=re.DOTALL) for document in
                         docs]
    # removal of selectors (<p>,</p>,...)
    documents_cleaned = [re.sub('<.*?>', ' ', document) for document in
                         documents_cleaned]
    # data editing...: punctuation, spacing, numbers and stopwords
    documents_cleaned = [re.sub(r'[^\w\s]', ' ', document) for document in
                         documents_cleaned]  # punctuation
    documents_cleaned = [re.sub('\s+', ' ', document) for document in
                         documents_cleaned]  # spaces, new line chars
    documents_cleaned = [re.sub('([0-9]+)', ' ', document) for document in
                         documents_cleaned]  # numbers
    # handle contextual synonyms
    for old, new in synonyms:
        documents_cleaned = [
            re.sub(old, new, document, flags=re.IGNORECASE) for document in
            documents_cleaned]
    # remove stopwords
    documents_cleaned = [[word for word in document.lower().split() if
                          word not in stopwords]
                         for document in documents_cleaned]
    # apply lemmatizer...
    documents_cleaned = [[get_lemma(word) for word in document] for
                         document in documents_cleaned]
    # make bigrams...
    bigram = gensim.models.Phrases(documents_cleaned, min_count=5,
                                   threshold=50)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    documents_cleaned = make_bigrams(bigram_mod, documents_cleaned)
    # filter out one char tokens like a, e, g (from e.g. and similar)...
    documents_cleaned = [[word
                          for word in document if len(word) > 1] for
                         document in documents_cleaned]
    # count word frequencies...
    frequency = collections.defaultdict(int)
    for document in documents_cleaned:
        for token in document:
            frequency[token] += 1
    # filter out words that appear less than 25 times
    documents_cleaned = [
        [token for token in document if frequency[token] >= 25] for
        document in documents_cleaned]
    return documents_cleaned


# read in data...
base_path = '../../Documents/TUM/BA/DataMining'
data = [
    read_csv_file(r'{}/qna-data/2013.csv'.format(base_path)),
    read_csv_file(r'{}/qna-data/2014.csv'.format(base_path)),
    read_csv_file(r'{}/qna-data/2015.csv'.format(base_path)),
    read_csv_file(r'{}/qna-data/2016.csv'.format(base_path)),
    read_csv_file(r'{}/qna-data/2017.csv'.format(base_path)),
    read_csv_file(r'{}/qna-data/2018.csv'.format(base_path)),
    read_csv_file(r'{}/qna-data/2019.csv'.format(base_path)),
    read_csv_file(r'{}/qna-data/2020.csv'.format(base_path))
]

# create time slice array for dtm from amount of questions in each year...
my_time_slices = []
for index, d in enumerate(data):
    my_time_slices.append(d[d.PostTypeId == 1].shape[0])
posts = pd.concat(data).reset_index(drop=True)
# persist data...
posts.to_pickle('{}/qna-data/posts-raw.pickle'.format(base_path))

# prepare input texts for dtm
questions = posts[posts.PostTypeId == 1]
input_texts = questions.Title + questions.Body
input_texts = input_texts.reset_index(drop=True)
input_texts.to_pickle('{}/qna-data/docs-raw.pickle'.format(base_path))

# define stop words
my_stopwords = nltk.corpus.stopwords.words('english')
additional_stopwords = ['able', 'also', 'another', 'explain', 'getting',
                        'help', 'hi', 'however', 'idea', 'im', 'issue',
                        'know', 'like', 'need', 'please', 'possible',
                        'problem', 'question', 'somebody', 'tried', 'try',
                        'trying', 'understand', 'use', 'using', 'want',
                        'way', 'without', 'working', 'would', 'yet',
                        'seems', 'see', 'unable']
my_stopwords.extend(additional_stopwords)
my_synonyms = [
    (' visual force ', ' visualforce '),
    (' sfdc ', ' salesforce com '),
    (' java script ', ' javascript '),
    (' vf ', ' visualforce '),
    (' js ', ' javascript '),
    (' sf ', ' salesforce '),
    (' front end ', ' frontend '),
    (' web service ', ' webservice '),
    (' vfpage ', ' visualforce page ')
]

# first take care of the special case for salesforce1
input_texts = [
    re.sub('salesforce1', 'salesforceone', document, flags=re.IGNORECASE)
    for document in input_texts]
# preprocess...
preprocessed_documents = preprocess_documents(docs=input_texts,
                                              stopwords=my_stopwords,
                                              synonyms=my_synonyms)
pd.Series(preprocessed_documents).to_pickle(
    '{}/qna-data/docs-preprocessed.pickle'.format(base_path))

# following code was run in the linux cluster...
DICT_PATH = 'ldaseq_dict'
CORPUS_PATH = 'ldaseq_corpus'
num_topics = 20
dictionary = gensim.corpora.Dictionary(preprocessed_documents)
corpus = [dictionary.doc2bow(text) for text in preprocessed_documents]
dictionary.save(DICT_PATH)
gensim.corpora.MmCorpus.serialize(CORPUS_PATH, corpus)
dynModel = gensim.models.LdaSeqModel(corpus=corpus, id2word=dictionary,
                                     time_slice=my_time_slices,
                                     num_topics=num_topics,
                                     chunksize=10000)
dynModel.save('QuestionModel_20Topics')
