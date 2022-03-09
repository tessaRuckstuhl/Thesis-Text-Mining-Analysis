import itertools
import re
from collections import Counter

import matplotlib.pylab as plt
import numpy as np
import pandas as pd

base_path = '../../Documents/TUM/BA/DataMining'
posts = pd.read_pickle('{}/qna-data/posts.pickle'.format(base_path))
questions = posts[posts.PostTypeId == 1]
answers = posts[posts.PostTypeId == 2]


# count and then plot the most used tags
def plot_tags():
    question_tags = questions['Tags'].str[1:-1].str.split('><')
    all_tags = itertools.chain.from_iterable(question_tags.array)
    count_dict = Counter(list(all_tags))
    lists = sorted(count_dict.items(), key=lambda kv: kv[1], reverse=True)[
            :50]
    x, y = zip(*lists)
    fix, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, y)
    ax.set_xlabel("Tags", fontsize=12)
    ax.set_ylabel("Tagged Questions", fontsize=12)
    ax.set_xticklabels(x, rotation=55, ha="right", fontsize=11)
    plt.show()


# calculate and plot evolution of top 10 tags
def plot_tags_evolution():
    top_tags = ['apex', 'visualforce', 'marketing-cloud',
                'lightning-aura-components',
                'trigger', 'lightning', 'soql', 'javascript',
                'lightning-web-components', 'unit-test']
    questions_evolution = pd.DataFrame()
    questions_evolution['CreationDate'] = pd.to_datetime(
        questions['CreationDate'], format='%Y-%m-%d %H:%M:%S')
    questions_evolution['Tags'] = questions['Tags'].str[1:-1].str.split(
        '><')
    monthly_questions_tags = \
        questions_evolution[['CreationDate', 'Tags']].groupby(
            questions_evolution.CreationDate.dt.to_period('M').rename(
                'Year'))[
            'Tags'].apply(lambda x: list(x))
    all_tags = monthly_questions_tags.apply(
        lambda x: np.concatenate(x)).apply(lambda x: Counter(x))
    dfs = []
    for tag in top_tags:
        tag_df = all_tags.apply(lambda x: x[tag])
        dfs.append(tag_df)
    for index, d in enumerate(dfs):
        ax = d.plot(figsize=(10, 5), linewidth=2, fontsize=12)
    plt.ylabel("Tagged Questions", fontsize=12)
    plt.xlabel("Year", fontsize=12)
    ax.legend(top_tags, bbox_to_anchor=(0, 1, 1, .102), loc=3,
              ncol=4, borderaxespad=0.,
              mode="expand", frameon=False, fontsize=12)
    plt.show()


# calculate and plot questions and answers per month
def plot_monthly_qna():
    questions['CreationDate'] = pd.to_datetime(
        questions['CreationDate'], format='%Y-%m-%d %H:%M:%S')
    answers['CreationDate'] = pd.to_datetime(
        answers['CreationDate'], format='%Y-%m-%d %H:%M:%S')
    questions_monthly = questions['CreationDate'].groupby(
        questions.CreationDate.dt.to_period('M').rename('Year')).agg(
        {'count'})
    answers_monthly = answers['CreationDate'].groupby(
        answers.CreationDate.dt.to_period('M').rename('Year')).agg(
        {'count'})
    ax = questions_monthly.plot(figsize=(10, 5), fontsize=12)
    answers_monthly.plot(ax=ax)
    plt.ylabel("Posts", fontsize=12)
    plt.xlabel("Year", fontsize=12)
    ax.legend(['Questions', 'Answers'], fontsize=12)
    plt.show()


# calculate and plot word counts of titles, question and answer bodies
def plot_word_counts():
    questions_cleaned = [re.sub('<.*?>', ' ', question)
                         for question in questions.Body]
    answers_cleaned = [re.sub('<.*?>', ' ', answer) for answer in
                       answers.Body]
    q_df = pd.Series([len(q.split()) for q in questions_cleaned])
    a_df = pd.Series([len(a.split()) for a in answers_cleaned])
    t_df = pd.Series([len(t.split()) for t in questions.Title])
    fig, axs = plt.subplots(3, figsize=(10, 6), sharex=True)
    axs[0].hist(t_df, bins=50, range=[0, 400])
    axs[0].axvline(t_df.median(), color='k', linestyle='dashed',
                   linewidth=1)
    axs[0].set_ylabel('Titles', fontsize=12)
    axs[1].hist(q_df, bins=50, range=[0, 400])
    axs[1].axvline(q_df.mean(), color='k', linestyle='dashed', linewidth=1)
    axs[1].set_ylabel('Questions', fontsize=12)
    axs[2].hist(a_df, bins=50, range=[0, 400])
    axs[2].axvline(a_df.mean(), color='k', linestyle='dashed', linewidth=1)
    axs[2].set_ylabel('Answers', fontsize=12)
    axs[2].set_xlabel('Words', fontsize=12)
    plt.show()


plot_tags()
plot_tags_evolution()
plot_word_counts()
plot_monthly_qna()
