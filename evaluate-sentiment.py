import csv

import matplotlib.pyplot as plt
import pandas as pd

base_path = '../../Documents/TUM/BA/DataMining'


# (1) calculate share of negative, neutral and positive classifications
# (2) plot sentiment
# (3) write data (abs and rel values) to excel
def plot_sentiment_data(topics):
    years = ['2013', '2014', '2015', '2016', '2017',
             '2018', '2019', '2020']
    fig, axs = plt.subplots(ncols=2)
    fig.text(0.5, 0, 'Years', ha='center', fontsize=14)
    fig.text(0.01, 0.5, 'Proportion', va='center',
             rotation='vertical', fontsize=14)
    for ax_idx, t in enumerate(topics):
        data_set = []
        table_data = []
        yearly_data = []
        print(ax_idx)
        for i in range(0, 8):
            # read in classified posts, files are divided by time slice
            data = pd.read_csv(
                '{}/sentiment/topic{}/output_chunk_{}.csv'.format(base_path,
                                                                  t, i),
                engine='python')
            yearly_data.append(len(data))
            # prepare data for plotting and create excel from data with stats
            data_set.append(
                data["Predicted"].value_counts(normalize=True).reindex(
                    ["positive", "neutral", "negative"]))
            c = data["Predicted"].value_counts().astype(str)
            p = data["Predicted"].value_counts(normalize=True).mul(100).round(
                1).astype(str)
            table_data.append(c + ", (" + p + ")")
        sets = pd.DataFrame(data_set, index=years)
        table = pd.DataFrame(table_data, index=years)
        table["Count"] = yearly_data
        table.to_csv(
            '{}/sentiment/Sentiment_{}_NewTable.csv'.format(base_path,
                                                            t),
            quoting=csv.QUOTE_NONE, sep="-")
        sets.plot(kind="area", stacked=True, figsize=(13, 6), ax=axs[ax_idx],
                  fontsize=12)
        axs[ax_idx].legend(loc="upper left", fontsize=12)
    plt.show()


plot_sentiment_data([1, 18])
