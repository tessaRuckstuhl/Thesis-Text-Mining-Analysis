import re

import pandas as pd

base_path = '../../Documents/TUM/BA/DataMining'
dominant_topics = pd.read_pickle(
    '{}/model/dominant_topics.pickle'.format(base_path))
my_time_slices = [5786, 11740, 13886, 16701, 16174, 13854, 15080, 15580]


# (1) cleanse data
# (2) write data (per time slice) to csv
def preprocess_for_classification(topic_id):
    dominant_topics[1] = [
        re.sub(r'<code.+?</code>', ' ', doc, flags=re.DOTALL) for doc in
        dominant_topics[1]]
    dominant_topics[1] = [re.sub('<.*?>', ' ', doc) for doc in
                          dominant_topics[1]]
    dominant_topics[1] = [re.sub('\s+', ' ', doc)
                          for doc in dominant_topics[1]]
    chunks = []
    count = 0
    for size in my_time_slices:
        chunks.append(dominant_topics[count:size + count])
        count += size
    for index, chunk in enumerate(chunks):
        chunk[chunk[0] == topic_id][1].to_csv(
            '{}/sentiment/topic_{}/input_chunk_{}.csv'.format(base_path,
                                                              topic_id,
                                                              index),
            sep=";", index=False,
            header=False)


# preprocess topics 1 and 18 for classification
for i in [1, 18]:
    preprocess_for_classification(i)
