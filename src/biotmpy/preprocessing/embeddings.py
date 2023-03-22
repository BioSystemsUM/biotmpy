from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models.wrappers import FastText
import numpy as np
from gensim.models import KeyedVectors
import codecs


def compute_embedding_matrix(config, embeddings_format, binary=True):
    """
    :param maxlen: We will cut reviews after n words
    :param training_samples: We will be training on x samples
    :param validation_samples: We will be validating on y samples
    :param max_nb_words:  We will only consider the top z words in the dataset
    :return:
    """
    embedding_path = config.embedding_path
    word_index = config.tokenizer.word_index
    embedding_dim = config.embedding_dim
    max_nb_words = config.max_nb_words
    print('Creating Embedding Matrix...')
    total_nb_words = 0
    words_not_found = []
    embeddings_index = {}
    if embeddings_format.lower() == 'glove':

        with open(embedding_path, encoding='utf-8') as f:
            for line in f:
                values = line.split()  # change to line.split (' ') on 840B glove
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        print('len word_index', len(word_index) + 1)
        print(max_nb_words)
        embedding_matrix = np.zeros((max_nb_words, embedding_dim))  # len(word_index)+1 or max_nb_words
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if i < max_nb_words:
                if embedding_vector is not None:
                    # Words not found in embedding index will be all-zeros.
                    embedding_matrix[i] = embedding_vector
                    total_nb_words += 1
                else:
                    words_not_found.append(word)
                    total_nb_words += 1

    elif embeddings_format.lower() == 'word2vec' or embeddings_format.lower() == 'fasttext':
        model_vec = KeyedVectors.load_word2vec_format(embedding_path, binary=binary)
        num_words = min(max_nb_words, len(word_index) + 1)
        embedding_matrix = np.zeros((num_words, embedding_dim))
        for word, i in word_index.items():
            total_nb_words += 1
            if i >= max_nb_words:
                continue
            if word in model_vec.vocab:
                embedding_vector = model_vec[word]
                embedding_vector = np.array(embedding_vector)
                if embedding_vector is not None:
                    # words not found in embedding index will be all-zeros.
                    embedding_matrix[i] = embedding_vector
            else:
                words_not_found.append(word)

    print('Embedding Matrix Created \n' + '-' * 24)
    null_words = np.sum(np.sum(embedding_matrix, axis=1) == 0)
    print('number of null word embeddings: {} in a total of {} words ({:.2%})'.format(null_words, total_nb_words,
                                                                                      null_words / total_nb_words))
    print('words not found:', len(words_not_found))
    if len(words_not_found) > 15:
        print("e.g. 10 words not found in the index : ", np.random.choice(words_not_found, 15))
    config.embedding_matrix = embedding_matrix
    return embedding_matrix


def convert_to_binary(embedding_path):
    """
    Here, it takes path to embedding text file provided by glove.
    :param embedding_path: takes path of the embedding which is in text format or any format other than binary.
    :return: a binary file of the given embeddings which takes a lot less time to load.
    """
    f = codecs.open(embedding_path + ".txt", 'r', encoding='utf-8')
    wv = []
    with codecs.open(embedding_path + ".vocab", "w", encoding='utf-8') as vocab_write:
        count = 0
        for line in f:
            if count == 0:
                pass
            else:
                splitlines = line.split()
                vocab_write.write(splitlines[0].strip())
                vocab_write.write("\n")
                wv.append([float(val) for val in splitlines[1:]])
            count += 1
    np.save(embedding_path + ".npy", np.array(wv))


def load_embeddings_binary(embeddings_path):
    """
    It loads embedding provided by glove which is saved as binary file. Loading of this model is
    about  second faster than that of loading of txt glove file as model.
    :param embeddings_path: path of glove file.
    :return: glove model
    """
    with codecs.open(embeddings_path + '.vocab', 'r', 'utf-8') as f_in:
        index2word = [line.strip() for line in f_in]
    wv = np.load(embeddings_path + '.npy')
    model = {}
    for i, w in enumerate(index2word):
        model[w] = wv[i]
    return model


def get_w2v(sentence, model):
    """
    :param sentence: inputs a single sentences whose word embedding is to be extracted.
    :param model: inputs glove model.
    :return: returns numpy array containing word embedding of all words    in input sentence.
    """
    return np.array([model.get(val, np.zeros(100)) for val in sentence.split()], dtype=np.float64)


def glove_embeddings_2(config):
    embedding_path = config.embedding_path
    convert_to_binary(embedding_path)
    return load_embeddings_binary(embedding_path)
