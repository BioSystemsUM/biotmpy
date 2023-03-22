import sys

sys.path.append('../')
from sklearn.feature_extraction.text import TfidfVectorizer
from yellowbrick.text import TSNEVisualizer, UMAPVisualizer
from yellowbrick.datasets import load_hobbies
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings

warnings.filterwarnings('ignore')
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebook
from collections import Counter
import os
import copy
from yellowbrick.style import set_palette
from yellowbrick.style.palettes import color_palette


def analysis_dataframe(x_train, y_train, dataset_name, set=None):
    """
    Creates a dataframe with the fulltext, word count and number of sentences of each document in the corpus. This dataframe is used for data analysis.

    :param x_train: list of Document objects
    :param y_train: list of Relevance objects
    :param dataset_name: name of the dataset
    :param set: set of the dataset (train, test, validation)

    :return: dataframe with the fulltext, word count and number of sentences of each document in the corpus
    """
    data = {'Document': x_train["Document"], 'fulltext': [], 'word_count': [], 'nmr_sentences': [], 'Label': []}
    for doc in x_train["Document"]:
        data['fulltext'].append(doc.fulltext_string)
        data['word_count'].append(len(doc.fulltext_tokens))
        data['nmr_sentences'].append(len(doc.sentences))
        data['Label'].append(y_train.loc[doc.id])
    df = pd.DataFrame(data)
    df['Label'] = df['Label'].replace(0, 'non-relevant').replace(1, 'relevant')
    path = os.path.join('data_analysis_plots/', dataset_name)
    if set:
        path = os.path.join(path, set)
    if not os.path.exists(path):
        os.makedirs(path)
    df.name = path
    return df


def corpus_visualization(analysis_df, dim_red='tsne', feature_ext='tfidf', file_name=None, title=None,
                         decompose_byint=50, random_state=None):
    """
    Creates a visualization of the corpus using the t-SNE or UMAP dimensionality reduction technique and the TF-IDF or Bag of Words feature extraction technique.

    :param analysis_df: dataframe with the fulltext, word count and number of sentences of each document in the corpus
    :param dim_red: dimensionality reduction technique (t-SNE or UMAP)
    :param feature_ext: feature extraction technique (TF-IDF or Bag of Words)
    :param file_name: name of the file to save the plot
    :param title: title of the plot
    :param decompose_byint: number of components to decompose the data
    :param random_state: random state for the dimensionality reduction technique

    :return: visualization of the corpus
    """
    name = analysis_df.name
    if title is None:
        if 'train' in name:
            name_title = 'Training'
        elif 'test' in name:
            name_title = 'Test'
        title = '{} Set'.format(name_title)
    if feature_ext.lower() == 'tfidf':
        extractor = TfidfVectorizer()
    elif feature_ext.lower() == 'bow':
        extractor = CountVectorizer()
    features = extractor.fit_transform(analysis_df.fulltext)
    if dim_red.lower() == 'tsne':
        visualizer = TSNEVisualizer(colormap=set_palette('sns_colorblind'), title=title,
                                    decompose_byint=decompose_byint, random_stateint=random_state, alphafloat=0.6)
    elif dim_red.lower() == 'umap':
        visualizer = UMAPVisualizer(colormap=set_palette('sns_colorblind'), title=title, random_stateint=random_state,
                                    alphafloat=0.6)
    # Create the visualizer and draw the vectors
    if file_name is None:
        if dim_red.lower() == 'tsne':
            file_name = 't-sne'
        elif dim_red.lower() == 'umap':
            file_name = 'umap'

    visualizer.fit(features, analysis_df.Label)

    custom_viz = visualizer.ax
    custom_viz.grid(False)
    custom_viz.set_title(title, fontsize=35, horizontalalignment='center')
    p = color_palette('sns_colorblind')
    custom_viz.legend(prop={'size': 22}, handlelength=1, edgecolor='gray', framealpha=0.5, frameon=True)

    custom_viz.set_xticklabels([])
    custom_viz.set_yticklabels([])
    custom_viz.figure.tight_layout()
    custom_viz.figure.savefig(analysis_df.name + '/' + file_name)
    custom_viz.figure.show()

    # visualizer.show(outpath=(name + '/' + file_name))


def plot_nmr_sentences(analysis_df, bins=100, xTitle='Number of Sentences', yTitle='Count',
                       title='Text Length Distribution', filename='text_len', fig_dim=(16, 5)):
    """
    Creates a plot of the distribution of the number of sentences per document in the corpus.

    :param analysis_df: dataframe with the fulltext, word count and number of sentences of each document in the corpus.
    :param bins: number of bins for the histogram. Default is 100
    :param xTitle: title of the x-axis. Default is 'Number of Sentences'
    :param yTitle: title of the y-axis. Default is 'Count'
    :param title: title of the plot. Default is 'Text Length Distribution'
    :param filename: name of the file to save the plot. Default is 'text_len'
    :param fig_dim: dimensions of the plot. Default is (16,5)

    :return: plot of the distribution of the number of sentences per document in the corpus
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_dim)
    fig.tight_layout()
    ax1.hist(analysis_df['nmr_sentences'], bins=bins)
    ax1.set_xlabel(xTitle, fontsize=15)
    ax1.set_ylabel(yTitle, fontsize=15)
    ax1.set_title(title, fontsize=20)
    ax2 = sns.violinplot(x="Label", y="nmr_sentences", data=analysis_df, palette=set_palette('colorblind'))
    ax2.set_ylabel('Number of sentences per document', fontsize=15)
    ax2.set_title(title, fontsize=20)
    ax2.set_xlabel('')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2)
    plt.savefig(analysis_df.name + '/' + filename)
    plt.show()


def plot_word_count(analysis_df, bins=100, xTitle='Number of words per Document', yTitle='Number of documents',
                    title='Word Count Distribution', filename='word_count', fig_dim=(16, 5), y_lim=None,
                    file_extension=None):
    """
    Creates a plot of the distribution of the number of words per document in the corpus.

    :param analysis_df: dataframe with the fulltext, word count and number of sentences of each document in the corpus
    :param bins: number of bins for the histogram. Default: 100
    :param xTitle: title of the x-axis. Default: 'Number of words per Document'
    :param yTitle: title of the y-axis. Default: 'Number of documents'
    :param title: title of the plot. Default: 'Word Count Distribution'
    :param filename: name of the file to save the plot. Default: 'word_count'
    :param fig_dim: dimensions of the plot. Default: (16,5)
    :param y_lim: limits of the y-axis. Default: None
    :param file_extension: extension of the file to save the plot. Default: None

    :return: plot of the distribution of the number of words per document in the corpus
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_dim)
    fig.tight_layout()
    ax1.hist(analysis_df['word_count'], bins=bins)
    ax1.set_title(title, fontsize=16)
    ax2 = sns.violinplot(x="Label", y="word_count", data=analysis_df, palette=set_palette('colorblind'))
    ax1.set_xlabel('Number of words', fontsize=13)
    ax1.set_ylabel('Number of Documents', fontsize=13)
    ax2.set_xlabel(xTitle)
    ax2.set_ylabel('Number of words per document', fontsize=13)
    ax2.set_title(title, fontsize=16)
    ax2.set_ylim(y_lim)
    ax1.tick_params(axis='both', which='major')
    ax2.tick_params(axis='both', which='major')
    plt.subplots_adjust(wspace=0.2)
    # plt.tight_layout()
    if not file_extension:
        plt.savefig(analysis_df.name + '/' + filename)
    else:
        plt.savefig(analysis_df.name + '/' + filename)
    plt.show()


def plot_words_per_sentence(analysis_df, bins=100, xTitle='Number of Words per Sentence', yTitle='Count',
                            title='Text Length Distribution', filename='text_len', threshold=None):
    """
    Creates a plot of the distribution of the number of words per sentence in the corpus.

    :param analysis_df: dataframe with the fulltext, word count and number of sentences of each document in the corpus.
    :param bins: number of bins for the histogram. Default is 100
    :param xTitle: title of the x-axis. Default is 'Number of Words per Sentence'
    :param yTitle: title of the y-axis. Default is 'Count'
    :param title: title of the plot. Default is 'Text Length Distribution'
    :param filename: name of the file to save the plot. Default is 'text_len'
    :param threshold: if set, only sentences with a number of words equal or higher than the threshold will be considered. Default is None

    :return: plot of the distribution of the number of words per sentence in the corpus
    """
    nmr_words_sentence = []
    nmr_sentences_threshold = 0
    nmr_sentences = 0
    for doc in analysis_df['Document']:
        for sentence in doc.sentences:
            if threshold and len(sentence.tokens) >= threshold:
                nmr_words_sentence.append(threshold)
                nmr_sentences_threshold += 1
            else:
                nmr_words_sentence.append(len(sentence.tokens))
            nmr_sentences += 1
    plt.hist(nmr_words_sentence, bins=bins)
    plt.xlabel(xTitle)
    plt.ylabel(yTitle)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(analysis_df.name + '/' + filename)
    plt.show()
    print('The lowest number of words in a sentence is:', min(nmr_words_sentence))
    print('The highest number of words in a sentece is:', max(nmr_words_sentence))
    print('Percentage of sentences with a number of words greater than or equal to {}: {:.2%}'.format(threshold,
                                                                                                      nmr_sentences_threshold / nmr_sentences))


def plot_labels_balance(analysis_df, filename='labels_balance'):
    """
    Creates a pie chart of the labels balance in the corpus.

    :param analysis_df: dataframe with the fulltext, word count and number of sentences of each document in the corpus.
    :param filename: name of the file to save the plot. Default is 'labels_balance'

    :return: pie chart of the labels balance in the corpus
    """
    d = {'relevant': [], 'non-relevant': []}
    d['relevant'] = analysis_df.loc[analysis_df['Label'] == 'relevant',].shape[0]
    d['non-relevant'] = analysis_df.loc[analysis_df['Label'] == 'non-relevant',].shape[0]
    plt.pie(d.values(), labels=d.keys(), autopct='%1.1f%%', shadow=True, startangle=90,
            colors=set_palette('colorblind'))
    plt.title(analysis_df.name[analysis_df.name.rfind("\\") + 1:] + ' set')
    plt.tight_layout()
    plt.savefig(analysis_df.name + '/' + filename)
    plt.show()


def plot_top_n_words(analysis_df, n=20, stop_words=None, file_name='top_words', n_grams=1, yTitle='Count',
                     title='Top 20 Words', fig_dim=(16, 6)):
    """
    Creates a plot of the top n words in the corpus.

    :param analysis_df: dataframe with the fulltext, word count and number of sentences of each document in the corpus.
    :param n: number of top words to plot. Default is 20
    :param stop_words: list of stop words to be removed from the corpus. Default is None
    :param file_name: name of the file to save the plot. Default is 'top_words'
    :param n_grams: number of words to be considered as a single word. Default is 1
    :param yTitle: title of the y-axis. Default is 'Count'
    :param title: title of the plot. Default is 'Top 20 Words'
    :param fig_dim: dimensions of the figure. Default is (16,6)

    :return: plot of the top n words in the corpus
    """
    name = analysis_df.name
    data = {}
    for lab in analysis_df['Label'].unique():
        df_lab = analysis_df.loc[analysis_df['Label'] == lab, 'fulltext']
        df_top = get_top_n_words(df_lab, n=n, stop_words=stop_words, n_grams=n_grams)
        data[lab] = df_top.groupby('fulltext').sum()['count'].sort_values(ascending=False)

    fig = plt.figure(figsize=fig_dim)
    fig.suptitle(title, fontsize=15, horizontalalignment='center')
    palette = sns.color_palette('colorblind', n)
    ax0 = fig.add_subplot(121)
    barlist1 = ax0.barh(data['non-relevant'].index.values, data['non-relevant'], color='w', edgecolor='gray')
    ax1 = fig.add_subplot(122)
    barlist2 = ax1.barh(data['relevant'].index.values, data['relevant'], color='w', edgecolor='gray')

    for i, word in enumerate(data['non-relevant'].index):
        for j, word_2 in enumerate(data['relevant'].index):
            if word == word_2:
                bar_color = palette.pop()
                barlist1[i].set_color(bar_color)
                barlist2[j].set_color(bar_color)

    ax0.set_title('non-relevant')
    ax1.set_title('relevant')

    ax0.set_facecolor('aliceblue')
    ax1.set_facecolor('honeydew')
    ax0.set_xlabel('counts')
    ax1.set_xlabel('counts')
    ax1.tick_params(axis='y', which='major')
    ax0.tick_params(axis='y', which='major')
    plt.subplots_adjust(wspace=0.6)
    # plt.tight_layout()
    plt.savefig(name + '/' + file_name)
    plt.show()

    words_nr = list(data['non-relevant'].index.values)
    words_r = list(data['relevant'].index.values)
    total_words = words_nr + words_r
    unique_words_nr = np.setdiff1d(words_nr, words_r, assume_unique=True)
    unique_words_r = np.setdiff1d(words_r, words_nr, assume_unique=True)
    print('Unique words: {} \n Non-Relevant: {} \n Relevant: {}'.format(len(unique_words_nr), unique_words_nr,
                                                                        unique_words_r))

    return unique_words_nr, unique_words_r


def LSA_topic_modelling(analysis_df, n_topics=6):
    """
    Performs topic modelling using Latent Semantic Analysis.

    :param analysis_df: dataframe with the fulltext, word count and number of sentences of each document in the corpus.
    :param n_topics: number of topics to be extracted. Default is 6

    :return: dataframe with the fulltext, word count and number of sentences of each document in the corpus.
    """
    reindexed_data = analysis_df['fulltext']
    tfidf_vectorizer = TfidfVectorizer(min_df=3, stop_words=set(stopwords.words('english')),
                                       ngram_range=(1, 3), analyzer="word")  # , use_idf=True, smooth_idf=True)
    reindexed_data = reindexed_data.values
    document_term_matrix = tfidf_vectorizer.fit_transform(reindexed_data)

    n_topics = n_topics
    lsa_model = TruncatedSVD(n_components=n_topics)
    lsa_topic_matrix = lsa_model.fit_transform(document_term_matrix)

    lsa_keys = get_keys(lsa_topic_matrix)
    lsa_categories, lsa_counts = keys_to_counts(lsa_keys)

    top_n_words_lsa = get_top_n_words_2(3, lsa_keys, document_term_matrix, tfidf_vectorizer, n_topics)

    for i in range(len(top_n_words_lsa)):
        print("Topic {}: ".format(i + 1), top_n_words_lsa[i])

    top_3_words = get_top_n_words_2(3, lsa_keys, document_term_matrix, tfidf_vectorizer, n_topics)
    labels = ['Topic {}: \n'.format(i) + top_3_words[i] for i in lsa_categories]

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.bar(lsa_categories, lsa_counts)
    ax.set_xticks(lsa_categories)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Number of documents')
    ax.set_title('LSA topic counts')
    plt.show()

    tsne_lsa_model = TSNE(n_components=2, perplexity=50, learning_rate=100,
                          n_iter=2000, verbose=1, random_state=0, angle=0.75)
    tsne_lsa_vectors = tsne_lsa_model.fit_transform(lsa_topic_matrix)

    colormap = np.array([
        "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
        "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
        "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
        "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"])
    colormap = colormap[:n_topics]

    top_3_words_lsa = get_top_n_words_2(3, lsa_keys, document_term_matrix, tfidf_vectorizer, n_topics)
    lsa_mean_topic_vectors = get_mean_topic_vectors(lsa_keys, tsne_lsa_vectors, n_topics)

    plot = figure(title="t-SNE Clustering of {} LSA Topics".format(n_topics), plot_width=700, plot_height=700)
    plot.scatter(x=tsne_lsa_vectors[:, 0], y=tsne_lsa_vectors[:, 1], color=colormap[lsa_keys])

    for t in range(n_topics):
        label = Label(x=lsa_mean_topic_vectors[t][0], y=lsa_mean_topic_vectors[t][1],
                      text=top_3_words_lsa[t], text_color=colormap[t])
        plot.add_layout(label)

    show(plot)


def get_top_n_words(corpus, n=None, stop_words=None, n_grams=1):
    """
    List the top n words in a vocabulary according to occurrence in a text corpus.

    :param corpus: list of documents
    :param n: number of words to return. Default is None
    :param stop_words: list of stop words to be removed. Default is None
    :param n_grams: number of words to be considered in each n-gram. Default is 1

    :return: dataframe with the top n words and their frequency
    """
    vec = CountVectorizer(stop_words=stop_words, ngram_range=(n_grams, n_grams)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    # for word, freq in words_freq[:n]:
    #     print(word, freq)
    df = pd.DataFrame(words_freq[:n], columns=['fulltext', 'count'])
    return df


def get_keys(topic_matrix):
    """
    returns a list of keys for a given topic matrix

    :param topic_matrix: matrix of topics

    :return: list of keys
    """
    keys = topic_matrix.argmax(axis=1).tolist()
    return keys


def keys_to_counts(keys):
    """
    returns a tuple of categories and counts for a given list of keys

    :param keys: list of keys

    :return: tuple of categories and counts
    """
    count_pairs = Counter(keys).items()
    categories = [pair[0] for pair in count_pairs]
    counts = [pair[1] for pair in count_pairs]
    return (categories, counts)


def get_top_n_words_2(n, keys, document_term_matrix, tfidf_vectorizer, n_topics):
    """
    returns a list of the top n words for each topic

    :param n: number of words to return
    :param keys: list of keys
    :param document_term_matrix: document term matrix
    :param tfidf_vectorizer: tfidf vectorizer
    :param n_topics: number of topics

    :return: list of top n words for each topic
    """
    top_word_indices = []
    for topic in range(n_topics):
        temp_vector_sum = 0
        for i in range(len(keys)):
            if keys[i] == topic:
                temp_vector_sum += document_term_matrix[i]
        temp_vector_sum = temp_vector_sum.toarray()
        top_n_word_indices = np.flip(np.argsort(temp_vector_sum)[0][-n:], 0)
        top_word_indices.append(top_n_word_indices)
    top_words = []
    for topic in top_word_indices:
        topic_words = []
        for index in topic:
            temp_word_vector = np.zeros((1, document_term_matrix.shape[1]))
            temp_word_vector[:, index] = 1
            the_word = tfidf_vectorizer.inverse_transform(temp_word_vector)[0][0]
            topic_words.append(the_word.encode('ascii').decode('utf-8'))
        top_words.append(" ".join(topic_words))
    return top_words


def get_mean_topic_vectors(keys, two_dim_vectors, n_topics):
    """
    returns a list of mean topic vectors

    :param keys: list of keys
    :param two_dim_vectors: list of two dimensional vectors
    :param n_topics: number of topics

    :return: list of mean topic vectors
    """
    mean_topic_vectors = []
    for t in range(n_topics):
        reviews_in_that_topic = []
        for i in range(len(keys)):
            if keys[i] == t:
                reviews_in_that_topic.append(two_dim_vectors[i])

        reviews_in_that_topic = np.vstack(reviews_in_that_topic)
        mean_review_in_that_topic = np.mean(reviews_in_that_topic, axis=0)
        mean_topic_vectors.append(mean_review_in_that_topic)
    return mean_topic_vectors
