# Thesis (A Text Mining Analysis)

This repository contains all Python Codes I implemented for my Bachelor's thesis "Examining Third-Party Developer Reactions to Changes to the Salesforce Platform Core: A Text Mining Analysis of the Salesforce Stack Exchange Community".

## Statistics

Some plots and statistics for better understanding of the data set.
## Dynamic Topic Model

Lda Sequence model was used for implementation, inspired by David M. Blei, John D. Lafferty: “Dynamic Topic Models”.

References:
- https://radimrehurek.com/gensim/models/ldaseqmodel.html

## Sentiment-Based Analysis

Polarity mining module (formerly Senti4SD) was used for implementation - An emotion-polarity classifier specifically trained on technical corpora from developers' communication channels.

### Setup

The easiest way to get set up is by running the toolkit for training custom sentiment and emotion classifiers from text in a docker container (https://github.com/collab-uniba/EMTk). 


References:
- https://github.com/collab-uniba/pySenti4SD
- https://github.com/collab-uniba/EMTk


