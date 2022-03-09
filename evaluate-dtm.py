import datetime
import time

import gensim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ts = time.time()
ts_str = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
base_path = '../../Documents/TUM/BA/DataMining'

# load model, dict, corpus and raw dochs
path_to_model = "{}/model/QuestionModel_20Topics".format(base_path)
my_dtm = gensim.models.LdaSeqModel.load(path_to_model)
my_dict = gensim.corpora.Dictionary.load(
    '{}/model/ldaseq_dict'.format(base_path))
my_corpus = gensim.corpora.MmCorpus(
    '{}/model/ldaseq_corpus'.format(base_path))
my_time_slices = [5786, 11740, 13886, 16701, 16174, 13854, 15080, 15580]
questions = pd.read_pickle('{}/qna-data/docs-raw.pickle'.format(base_path))
num_topics = 20
# result from function get_dominant_topic_for_doc()
dominant_topics = pd.read_pickle(
    '{}/model/dominant_topics.pickle'.format(base_path))


# writes topic model results to excel
def write_results_to_excel():
    num_top_terms = 10
    with pd.ExcelWriter(
            '{}/output/Formatted_TopicEvolution_Probs_{}_{}{}'.format(
                base_path, num_topics, ts_str,
                '.xlsx')) as writer:
        for i in range(0, num_topics):
            topic_output = pd.DataFrame()
            topic_times = my_dtm.print_topic_times(topic=i,
                                                   top_terms=num_top_terms)
            for j, topic in enumerate(topic_times):
                s = pd.Series("{}, ({})".format(w[0], w[1]) for w in topic)
                topic_output[str(2013 + j)] = s
            topic_output.to_excel(writer, sheet_name="Topic {}".format(i))


# (1) maps the most dominant topic to each document and
# (2) saves result to pickle
def get_dominant_topic_for_doc():
    mapping = questions.apply(lambda x: pd.Series(
        [np.where(my_dtm.doc_topics(x.name) == np.amax(
            my_dtm.doc_topics(x.name)))[0][0], x.input_texts]), axis=1)
    mapping.to_pickle('{}/model/dominant_topics.pickle'.format(base_path))


# (1) counts topic shares for each time slice and each topic
# (2) plots shares for every single topic
def plot_topic_shares_per_topic():
    chunks = []
    count = 0
    for size in my_time_slices:
        chunks.append(dominant_topics[0][
                      count:size + count].value_counts().sort_index())
        count += size
    data_stacked = pd.DataFrame(chunks,
                                index=[2013, 2014, 2015, 2016,
                                       2017, 2018, 2019, 2020])
    fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(10, 12), sharex=True,
                             sharey=True)
    plt.subplots_adjust(hspace=0.2, wspace=0.1, bottom=0.05, top=0.98,
                        left=0.075, right=0.975)
    for row, ax in zip(data_stacked.T.index, axes.flatten()):
        x = data_stacked.T.loc[row].index
        p = np.poly1d(np.polyfit(x, data_stacked.T.loc[row].values, 1))
        ax.plot(p(x), 'r--')
        ax.plot(data_stacked.T.loc[row].values, marker='.')
        ax.set_title(row, fontsize=12)
        ax.set_ylim([0, 2000])
        ax.set_xticks(range(data_stacked.T.shape[1]))
        ax.set_xticklabels(list(data_stacked.T.columns), rotation=45,
                           fontsize=8)
    fig.text(0.5, 0, 'Years', ha='center', fontsize=14)
    fig.text(0.01, 0.5, 'Questions', va='center',
             rotation='vertical', fontsize=14)
    plt.show()


# (1) get topic assignments of a specific document
# (2) plot topic assignment
def plot_topic_mixtures(doc_id):
    topic_dist = pd.Series(my_dtm.doc_topics(doc_id))
    ax = topic_dist.plot(kind='bar', rot=0, figsize=(10, 2))
    plt.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.27)
    ax.set_xlabel("Topic Number", fontsize=12)
    ax.set_ylabel('Topic Proportion', fontsize=12)
    plt.show()


# (1) get top ten terms from first and last time slice
# (2) get probs for every single term in every single time slice
# (3) create dict for further usage
def create_topwords_dict():
    dicts = []
    for i in range(0, num_topics):
        topwords_start = my_dtm.print_topic(topic=i, time=0, top_terms=10)
        topwords_end = my_dtm.print_topic(topic=i, time=7, top_terms=10)
        topwords = []
        # create top words from 2013 and 2020
        for index, (w, p) in enumerate(topwords_start):
            if not w in topwords:
                topwords.append(w)
        for index, (w, p) in enumerate(topwords_end):
            if not w in topwords:
                topwords.append(w)
        dicts_top_term = []
        for j, w in enumerate(topwords):
            probs = []
            topwords_times = my_dtm.print_topic_times(topic=i,
                                                      top_terms=400)
            [[probs.append(tuple((2013 + index, prob))) for (word, prob) in
              tw_slice if w == word]
             for (index, tw_slice) in enumerate(topwords_times)]
            dicts_top_term.append({w: probs})
        dicts.append(dicts_top_term)
    return dicts


colors = list(plt.cm.tab10(np.arange(10))) + ["navy", "darkgreen",
                                              "teal", "magenta",
                                              "goldenrod"]


# plots the topic term trends of four topics in one plot
def plot_topic_evolution(dicts, offset=0):
    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(10, 14))
    plt.subplots_adjust(hspace=0.6, wspace=0.15, bottom=0.05, top=0.95,
                        left=0.08, right=0.95)
    for index, topic_dicts in enumerate(dicts[offset:offset + 4]):
        for i, dic in enumerate(topic_dicts):
            for (k, v) in dic.items():
                axs[index, 0].plot([x[0] for x in v], [x[1] for x in v],
                                   label=k, marker='o', color=colors[i])
                axs[index, 0].legend(bbox_to_anchor=(0, 1, 2, .102), loc=3,
                                     ncol=8, borderaxespad=0.,
                                     mode="expand", frameon=False)
                axs[index, 0].set_title("Topic " + str(
                    index + offset) + " - A (Full range)\n\n")
    for index, topic_dicts in enumerate(dicts[offset:offset + 4]):
        for i, dic in enumerate(topic_dicts):
            for (k, v) in dic.items():
                axs[index, 1].plot([x[0] for x in v], [x[1] for x in v],
                                   label=k, marker='o', color=colors[i])
                # zoom in...
                axs[index, 1].set_ylim([0, 0.04])
                axs[index, 1].set_title("Topic " + str(
                    index + offset) + " - B (Selected range)\n\n")
    fig.text(0.5, 0, 'Years', ha='center', fontsize=14)
    fig.text(0.01, 0.5, 'Probability', va='center',
             rotation='vertical', fontsize=14)
    plt.show()


# plots the topic term trends of a single topic
def plot_single_topic_evolution(topic_dict, idx):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    plt.subplots_adjust(hspace=0.6, wspace=0.155, bottom=0.1, top=0.8,
                        left=0.075, right=0.975)
    for index, topic_dicts in enumerate([topic_dict[idx]]):
        for i, dic in enumerate(topic_dicts):
            for (k, v) in dic.items():
                axs[0].plot([x[0] for x in v], [x[1] for x in v], label=k,
                            marker='o', color=colors[i])
                axs[0].legend(bbox_to_anchor=(0, 1.03, 2, .102), loc=3,
                              ncol=8, borderaxespad=0., mode="expand",
                              frameon=False)
                axs[0].set_title(
                    "Topic " + str(idx) + " - A (Full range)\n\n\n")

    for index, topic_dicts in enumerate([topic_dict[idx]]):
        for i, dic in enumerate(topic_dicts):
            for (k, v) in dic.items():
                axs[1].plot([x[0] for x in v], [x[1] for x in v], label=k,
                            marker='o', color=colors[i])
                # zoom in...
                axs[1].set_ylim([0, 0.04])
                axs[1].set_title(
                    "Topic " + str(idx) + " - B (Selected range)\n\n\n")
    fig.text(0.5, 0, 'Years', ha='center', fontsize=14)
    fig.text(0.01, 0.5, 'Probability', va='center',
             rotation='vertical', fontsize=14)
    plt.show()


# (1) round probabilities of every term
# (2) create excel sheet from dict for usage in thesis
def create_term_probs_df(t, dicts):
    dfs = []
    print(dict)
    for i, dic in enumerate(dicts[t]):
        for (k, v) in dic.items():
            df = pd.DataFrame([round(x[1], 3) for x in v],
                              index=(
                                  "2013", "2014", "2015", "2016", "2017",
                                  "2018", "2019", "2020")).transpose()
            df.insert(0, 'Term', k)
            dfs.append(df)
    pd.concat(dfs).to_excel(
        '{}/output/terms/term_topic_{}_probs_table.xlsx'.format(base_path,
                                                                t))
    return pd.concat(dfs)


ids = [100497, 100046, 100310, 100023, 11803, 100206, 100385]
for id in ids:
    plot_topic_mixtures(id)

plot_topic_shares_per_topic()
write_results_to_excel()
dicts = create_topwords_dict()
plot_single_topic_evolution(dicts, 18)
plot_single_topic_evolution(dicts, 1)
plot_topic_evolution(dicts, offset=0)
plot_topic_evolution(dicts, offset=4)
plot_topic_evolution(dicts, offset=8)
plot_topic_evolution(dicts, offset=12)
plot_topic_evolution(dicts, offset=16)
