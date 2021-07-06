import sys
sys.path.append('../')
import pandas as pd
from collections import Counter
from .text_summarization import generate_summary
# from NLP import *
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import wordpunct_tokenize
import string
import pickle
import re
import math
from .term_lists.mutation_lists import strong_mutation_list, weak_mutation_list, complete_mutation_list, \
                                        		interactions_list, impact_list, good_indicators_list, bad_indicators_list

def generate_features_dr(dataframe, features_list=None):
    documents = list(dataframe['Document'])
    if features_list is not None:
        for feature in features_list:
            dataframe = method_by_feature[feature](dataframe, documents)
    else:
        dataframe = generate_all_features(dataframe, documents)

    dataframe = dataframe.drop('Document', axis = 1)
    return dataframe


def generate_all_features(dataframe, documents):
    import time
    start_time = time.time()
    dataframe = nmr_tokens(dataframe, documents)
    print("1 --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    dataframe = sentence_size_stats(dataframe, documents)
    print("2 --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    dataframe = nmr_vowels(dataframe, documents)
    print("3 --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    dataframe = title_sentence_size(dataframe, documents)
    print("4 --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    dataframe = distinct_stems(dataframe, documents)
    print("5 --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    dataframe = nmr_unusual_words(dataframe, documents)
    print("6 --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    dataframe = mutation_strong_weak(dataframe, documents)
    print("7 --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    dataframe = mutation_interaction_impact(dataframe, documents)
    print("8 --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    dataframe = mutation_interaction_impact_v2(dataframe, documents)
    print("9 --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    dataframe = mutation_interaction_impact_sentences(dataframe, documents)
    print("10 --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    dataframe = mutation_interaction_impact_sentences_v2(dataframe, documents)
    print("11 --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    dataframe = mutation_interaction_impact_title(dataframe, documents)
    print("12 --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    dataframe = nmr_protein_names_n_ids(dataframe, documents)
    print("13 --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    dataframe = nmr_protein_tokens(dataframe, documents)
    print("14 --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    dataframe = mutation_interaction_impact_summary(dataframe, documents)
    print("15 --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    dataframe = mutation_interaction_impact_summary_v2(dataframe, documents)
    print("16 --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    dataframe = strong_weak_mutation_summary(dataframe, documents)
    print("17 --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    dataframe = nmr_stopwords(dataframe, documents)
    print("18 --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    dataframe = indicators(dataframe, documents)
    print("19 --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    dataframe = fulltext_size(dataframe, documents)
    print("20 --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    dataframe = tokens_size_stats(dataframe, documents)
    print("21 --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    dataframe = punctuation(dataframe, documents)
    print("22 --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    dataframe = capital_letters(dataframe, documents)
    print("23 --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    dataframe = POS_counter(dataframe, documents)
    print("24 --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    dataframe = mut_prot_sent(dataframe, documents)
    print("25 --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    dataframe = nmr_protein_tokens_lower(dataframe,documents)
    print("26 --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    dataframe = nmr_protein_tokens_v2(dataframe, documents)
    print("27 --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    dataframe = tfidf(dataframe, documents)
    print("tfidf --- %s seconds ---" % (time.time() - start_time))
    return dataframe


def generate_features_sentence_level(dataframe, documents):
    dataframe = sentence_size_stats(dataframe, documents)
    dataframe = title_sentence_size(dataframe, documents)
    dataframe = mutation_interaction_impact_summary(dataframe, documents)
    dataframe = mutation_interaction_impact_summary_v2(dataframe, documents)
    dataframe = strong_weak_mutation_summary(dataframe, documents)
    dataframe = fulltext_size(dataframe, documents)
    return dataframe


def generate_features_word_level(dataframe, documents):
    dataframe = nmr_tokens(dataframe, documents)
    dataframe = nmr_vowels(dataframe, documents)
    dataframe = distinct_stems(dataframe, documents)
    dataframe = nmr_unusual_words(dataframe, documents)
    dataframe = mutation_strong_weak(dataframe, documents)
    dataframe = mutation_interaction_impact(dataframe, documents)
    dataframe = mutation_interaction_impact_v2(dataframe, documents)
    dataframe = mutation_interaction_impact_sentences(dataframe, documents)
    dataframe = mutation_interaction_impact_sentences_v2(dataframe, documents)
    dataframe = mutation_interaction_impact_title(dataframe, documents)
    dataframe = nmr_protein_names_n_ids(dataframe, documents)
    dataframe = nmr_protein_tokens(dataframe, documents)
    dataframe = nmr_stopwords(dataframe, documents)
    dataframe = indicators(dataframe, documents)
    dataframe = tokens_size_stats(dataframe, documents)
    dataframe = punctuation(dataframe, documents)
    dataframe = capital_letters(dataframe, documents)
    dataframe = POS_counter(dataframe, documents)
    return dataframe


def get_summary(documents): #SUMMARIZATION
    res = []
    for d in documents:
        summary = generate_summary(d.fulltext_string)
        res.append(summary)
    return res


def insert_column(dataframe, new_column, column_name, dummies=False, label_prefix=None):
    indexes = list(dataframe.index.values)
    dataframe[column_name] = pd.Series(new_column, index=indexes)
    if dummies:
        df_dummies = pd.get_dummies(dataframe[column_name], prefix = label_prefix)
        dataframe = pd.concat([dataframe.drop(column_name, axis = 1), df_dummies], axis = 1)
    return dataframe


def nmr_tokens(dataframe, documents):
    res_title, res_abstract, res_total = [], [], []
    for d in documents:
        res_title.append(len(d.title_tokens))
        res_abstract.append(len(d.abstract_tokens))
        res_total.append(len(d.fulltext_tokens))
    dataframe = insert_column(dataframe, res_title, 'Title_Nmr_Tokens')
    dataframe = insert_column(dataframe, res_abstract, 'Abstract_Nmr_Tokens')
    dataframe = insert_column(dataframe, res_total, 'Total_Nmr_Tokens')
    return dataframe


def sentence_size_stats(dataframe, documents):
    res_mean, res_max, res_min = [], [], []
    for d in documents:
        max_nmr, min_nmr = len(d.sentences[0].string), len(d.sentences[0].string)
        total = 0
        for sentence in d.sentences:
            if len(sentence.string) > max_nmr:
                max_nmr = len(sentence.string)
            if len(sentence.string) < min_nmr:
                min_nmr = len(sentence.string)
            total += len(sentence.string)
        if len(d.sentences) != 0:
            res_mean.append(total / len(d.sentences))
        res_max.append(max_nmr)
        res_min.append(min_nmr)
    dataframe = insert_column(dataframe, res_mean, 'Mean_sentence_size')
    dataframe = insert_column(dataframe, res_max, 'Max_sentence_size')
    dataframe = insert_column(dataframe, res_min, 'Min_sentence_size')
    return dataframe


def title_sentence_size(dataframe, documents):
    res = []
    for d in documents:
        res.append(len(d.title_string))
    dataframe = insert_column(dataframe, res, 'Title_sentence_Size')
    return dataframe


def nmr_vowels(dataframe, documents):
    res_vowels = []
    for d in documents:
        sum_vowels = 0
        for letter in d.fulltext_string.lower():
            if (letter == 'a' or letter == 'e' or letter == 'i' or letter == 'o' or letter == 'u'):
                sum_vowels += 1
        res_vowels.append(sum_vowels)
    dataframe = insert_column(dataframe, res_vowels, 'Nmr_Vowels')
    return dataframe


def distinct_stems(dataframe, documents):
    res_stems = []
    stemmer = PorterStemmer()
    for d in documents:
        distinct_stems = []
        for token in d.fulltext_tokens:
            stem = stemmer.stem(token.string)
            if stem not in distinct_stems:
                distinct_stems.append(stem)
        res_stems.append(len(distinct_stems))
    dataframe = insert_column(dataframe, res_stems, 'Nmr_Distinct_Stems')
    return dataframe


def nmr_unusual_words(dataframe, documents):
    res = []
    stop_words = set(stopwords.words('english'))
    punctuation_string = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    for d in documents:
        nmr_unusual_words = 0
        for token in d.fulltext_tokens:
            if token.string.lower() not in stop_words and token.string not in punctuation_string:
                nmr_unusual_words +=1
        res.append(nmr_unusual_words)
    dataframe = insert_column(dataframe, res, 'Nmr_Unusual_Words')
    return dataframe


def mutation_strong_weak(dataframe, documents):
    res = []
    for d in documents:
        score = 0
        for sentence in d.sentences:
            for str_mut in strong_mutation_list:
                if str_mut in sentence.string.lower():
                    score += 3
            for weak_mut in weak_mutation_list:
                if weak_mut in sentence.string.lower():
                    score += 1
        res.append(score)
    dataframe = insert_column(dataframe, res, 'Mutation_List')
    return dataframe


def mutation_interaction_impact(dataframe, documents):
    res = []
    for d in documents:
        score = 0
        for mut in complete_mutation_list:
            if mut in d.fulltext_string.lower():
                score += 1
                break
        for inter in interactions_list:
            if inter in d.fulltext_string.lower():
                score += 1
                break
        for imp in impact_list:
            if imp in d.fulltext_string.lower():
                score +=1
                break
        if score == 0:
            res.append(-3)
        else:
            res.append(score)
    dataframe = insert_column(dataframe, res, 'Mut_Int_Imp')
    return dataframe


def mutation_interaction_impact_v2(dataframe, documents):
    res = []
    for d in documents:
        score = 0
        for mut in complete_mutation_list:
            if mut in d.fulltext_string.lower():
                score += 1
        for inter in interactions_list:
            if inter in d.fulltext_string.lower():
                score += 1
        for imp in impact_list:
            if imp in d.fulltext_string.lower():
                score +=1
        if score == 0:
            res.append(-1)
        else:
            res.append(score)
    dataframe = insert_column(dataframe, res, 'Mut_Int_Imp_v2')
    return dataframe


def mutation_interaction_impact_sentences(dataframe, documents):
    res = []
    for d in documents:
        total_score = 0
        for sentence in d.sentences:
            sent_score = 0
            for mut in complete_mutation_list:
                if mut in sentence.string.lower():
                    sent_score += 1
                    break
            for inter in interactions_list:
                if inter in sentence.string.lower():
                    sent_score += 1
                    break
            for imp in impact_list:
                if imp in sentence.string.lower():
                    sent_score += 1
                    break
            if sent_score == 3:
                total_score += 1
        res.append(total_score)
    dataframe = insert_column(dataframe, res, 'MII_Sentences')
    return dataframe


def mutation_interaction_impact_sentences_v2(dataframe, documents):
    res = []
    for d in documents:
        total_score = 0
        for sentence in d.sentences:
            sent_score = 0
            for mut in complete_mutation_list:
                if mut in sentence.string.lower():
                    sent_score += 1
            for inter in interactions_list:
                if inter in sentence.string.lower():
                    sent_score += 1
            for imp in impact_list:
                if imp in sentence.string.lower():
                    sent_score += 1
            total_score += sent_score
        res.append(total_score)
    dataframe = insert_column(dataframe, res, 'MII_Sentences_v2')
    return dataframe


def mutation_interaction_impact_title(dataframe, documents):
    res = []
    for d in documents:
        score = 0
        for mut in complete_mutation_list:
            if mut in d.title_string.lower():
                score += 1
        for inter in interactions_list:
            if inter in d.title_string.lower():
                score += 1
        for imp in impact_list:
            if imp in d.title_string.lower():
                score += 1
        res.append(score)
    dataframe = insert_column(dataframe, res, 'MII_Sent_Title')
    return dataframe


def nmr_protein_names_n_ids(dataframe, documents):
    res = []
    protein_full_names_list = get_protein_names_n_ids('../pipelines/term_lists/uniprot.tab')
    for d in documents:
        score = 0
        for sentence in d.sentences:
            for full_name in protein_full_names_list:
                if full_name in sentence.string:
                    score += 1
        res.append(score)
    dataframe = insert_column(dataframe, res, 'Nmr_Prot_Full_Name')
    return dataframe


def nmr_protein_tokens(dataframe, documents):
    res = []
    prot_tokens = get_tokens_protein_names_n_ids('../pipelines/term_lists/uniprot.tab')
    for d in documents:
        score = 0
        for sentence in d.sentences:
            for token in prot_tokens:
                if token in sentence.string:
                    score += 1
        res.append(score)
    dataframe = insert_column(dataframe, res, 'Nmr_Prot_Tokens')
    return dataframe


def nmr_protein_tokens_v2(dataframe, documents):
    res = []
    prot_tokens = get_tokens_protein_names_n_ids('../pipelines/term_lists/uniprot.tab')
    for d in documents:
        score = 0
        for token in d.fulltext_tokens:
            if token.string in prot_tokens:
                    score += 1
        res.append(score)
    dataframe = insert_column(dataframe, res, 'Nmr_Prot_Tokens_v2')
    return dataframe


def nmr_protein_tokens_lower(dataframe, documents):
    res = []
    prot_tokens = get_tokens_protein_names_n_ids('../pipelines/term_lists/uniprot.tab')
    for d in documents:
        score = 0
        for sentence in d.sentences:
            for token in prot_tokens:
                if token.lower() in sentence.string.lower():
                    score += 1
        res.append(score)
    dataframe = insert_column(dataframe, res, 'Nmr_Prot_Tokens_lower')
    return dataframe


def mutation_interaction_impact_summary(dataframe, documents):
    res = []
    docs_summary = get_summary(documents)
    for doc_top_2phrases in docs_summary:
        score = 0
        for mut in complete_mutation_list:
            if mut in doc_top_2phrases.lower():
                score += 1
        for inter in interactions_list:
            if inter in doc_top_2phrases.lower():
                score += 1
        for imp in impact_list:
            if imp in doc_top_2phrases.lower():
                score += 1
        if score == 0:
            res.append(-1)
        else:
            res.append(score)
    dataframe = insert_column(dataframe, res, 'MII_Summary')
    return dataframe


def mutation_interaction_impact_summary_v2(dataframe, documents):
    res = []
    docs_summary = get_summary(documents)
    for doc_top_2phrases in docs_summary:
        score = 0
        for mut in complete_mutation_list:
            if mut in doc_top_2phrases.lower():
                score += 1
                break
        for inter in interactions_list:
            if inter in doc_top_2phrases.lower():
                score += 1
                break
        for imp in impact_list:
            if imp in doc_top_2phrases.lower():
                score += 1
                break
        if score == 0:
            res.append(-3)
        elif score == 3:
            res.append(3)
        else:
            res.append(1)
    dataframe = insert_column(dataframe, res, 'MII_Summary_v2')
    return dataframe


def strong_weak_mutation_summary(dataframe, documents):
    res = []
    docs_summary = get_summary(documents)
    for doc_top_2phrases in docs_summary:
        score = 0
        for s in strong_mutation_list:
            if s in doc_top_2phrases.lower():
                score += 3
        for w in weak_mutation_list:
            if w in doc_top_2phrases.lower():
                score += 1
        if score == 0:
            res.append(-1)
        else:
            res.append(score)
    dataframe = insert_column(dataframe, res, 'Strong_Weak_Summary')
    return dataframe


def nmr_stopwords(dataframe, documents):
    res = []
    stopWords = set(stopwords.words('english'))
    for d in documents:
        nmr_stopwords = 0
        for token in d.fulltext_tokens:
            if token.string.lower() in stopWords:
                nmr_stopwords += 1
        res.append(nmr_stopwords)
    dataframe = insert_column(dataframe, res, 'Nmr_Stopwords')
    return dataframe


def indicators(dataframe, documents):
    good_res, bad_res, global_score_res = [], [], []
    for d in documents:
        good_score, bad_score, global_score = 0, 0, 0
        for good in good_indicators_list:
            if good in d.fulltext_string.lower():
                good_score += 1
                global_score +=1
        for bad in bad_indicators_list:
            if bad in d.fulltext_string.lower():
                bad_score += 1
                global_score -= 1
        good_res.append(good_score)
        bad_res.append(bad_score)
        global_score_res.append(global_score)
    dataframe = insert_column(dataframe, good_res, 'Good_Indicators')
    dataframe = insert_column(dataframe, bad_res, 'Bad_Indicators')
    dataframe = insert_column(dataframe, global_score_res, 'Indicators')
    return dataframe


def fulltext_size(dataframe, documents):
    res = []
    for d in documents:
        res.append(len(d.fulltext_string))
    dataframe = insert_column(dataframe, res, 'FullText_Size')
    return dataframe


def tokens_size_stats(dataframe, documents):
    mean_res, max_res = [], []
    for d in documents:
        max_score, count = 0, 0
        for token in d.fulltext_tokens:
            if len(token.string) > max_score:
                max_score = len(token.string)
            count += len(token.string)
        if len(d.sentences) != 0:
            mean_res.append(count / len(d.fulltext_tokens))
        max_res.append(max_score)
    dataframe = insert_column(dataframe, mean_res, 'Mean_tokens_size')
    dataframe = insert_column(dataframe, max_res, 'Max_tokens_size')
    return dataframe


def capital_letters(dataframe, documents):
    res = []
    for d in documents:
        nmr_capitalLetters = 0
        for letter in d.fulltext_string:
            if letter.isupper():
                nmr_capitalLetters += 1
        res.append(nmr_capitalLetters)
    dataframe = insert_column(dataframe, res, 'Nmr_CapitalLetters')
    return dataframe


def punctuation(dataframe, documents):
    punct_res = []
    count = lambda l1, l2: sum([1 for x in l1 if x in l2])
    punctuation_string = set(string.punctuation)
    for d in documents:
        s = d.fulltext_string
        res = count(s, punctuation_string)
        punct_res.append(res)
    dataframe = insert_column(dataframe, punct_res, 'Punctuation')
    return dataframe


def POS_counter(dataframe, documents):
    nouns, pnouns, verbs, adjectives, adverbs, ppronouns = [], [], [], [], [], []
    for d in documents:
        tagged_words = pos_tag(wordpunct_tokenize(d.fulltext_string))
        count = Counter([j for i, j in tagged_words])
        nouns.append(count['NN'])
        pnouns.append(count['NNP'])
        all_verbs = count['VB'] + count['VBD'] + count['VBG'] + count['VBN'] + count['VBP'] + count['VBZ']
        verbs.append(int(all_verbs))
        adjectives.append(count['JJ'])
        adverbs.append(count['RB'])
        ppronouns.append(count['PRP'])
    dataframe = insert_column(dataframe, nouns, 'Nouns')
    dataframe = insert_column(dataframe, pnouns, 'Proper_nouns')
    dataframe = insert_column(dataframe, verbs, 'Verbs')
    dataframe = insert_column(dataframe, adjectives, 'Adjectives')
    dataframe = insert_column(dataframe, adverbs, 'Adverbs')
    dataframe = insert_column(dataframe, ppronouns, 'Personal_Pronouns')
    return dataframe


def mut_prot_sent(dataframe, documents):
    res = []
    prot_tokens = get_tokens_protein_names_n_ids('../pipelines/term_lists/uniprot.tab')
    for d in documents:
        globalscore = 0
        for sentence in d.sentences:
            sentscore = 0
            for token in prot_tokens:
                if token.lower() in sentence.string.lower():
                    sentscore += 1
                    break
            for m in complete_mutation_list:
                if m in sentence.string.lower():
                    sentscore += 1
                    break
            if sentscore == 2:
                globalscore += 1
        if globalscore == 0:
            res.append(-1)
        else:
            res.append(globalscore)
    dataframe = insert_column(dataframe, res, 'Mut_Prot_sent')
    return dataframe


def compute_tf(wordDict_doc, bow_doc):
    tf_dict = {}
    bow_count = len(bow_doc)
    for word, count in wordDict_doc.items():
        tf_dict[word] = count / float(bow_count)
    return tf_dict


def compute_idf(doc_list):
    N = len(doc_list)

    #counts the number of documents that contain a word w
    idf_dict = dict.fromkeys(doc_list[0].keys(), 0)
    for doc in doc_list:
        for word, val in doc.items():
            if val > 0:
                idf_dict[word] += 1

    #divide N by denominator aboce, take the log of that
    for word, val in idf_dict.items():
        idf_dict[word] = math.log(N / float(val))

    return idf_dict


def compute_tfidf(tf_bow_doc, idfs):
    tfidf = {}
    for word, val in tf_bow_doc.items():
        tfidf[word] = val * idfs[word]
    return tfidf


def tfidf(dataframe, documents):
    list_bows_doc = []
    for d in documents:
        tokens_without_punctuation = []
        for token in d.fulltext_tokens:
            s = token.string.translate(str.maketrans('', '', string.punctuation))
            tokens_without_punctuation.append(s.encode("utf-8"))
        list_bows_doc.append(tokens_without_punctuation)
    bow_total = set().union(*list_bows_doc)   #bag-of-words
    docs_dict = {}
    index_list = []
    for d in documents:
        index_list.append(d.id)
        doc_dict = dict.fromkeys(bow_total, 0)
        docs_dict[d.id] = doc_dict
    for d in documents:
        for tk in d.fulltext_tokens:
            s = tk.string.translate(str.maketrans('', '', string.punctuation)).encode("utf-8")
            docs_dict[d.id][s] += 1
    tf_bow, doc_list = [], []
    for i, k in enumerate(docs_dict.keys()):
        doc_list.append(docs_dict[k])
        tf_bow.append(compute_tf(docs_dict[k], list_bows_doc[i]))
    idfs = compute_idf(doc_list)
    tfidf_bow = []
    for tf_res in tf_bow:
        tfidf_bow.append(compute_tfidf(tf_res, idfs))
    df_tfidf = pd.DataFrame(tfidf_bow, index= index_list)
    df_tfidf = df_tfidf.drop(columns=[b'']).fillna(0)
    df_res = pd.concat([dataframe, df_tfidf], axis=1)
    return df_res


def get_protein_names_n_ids(uniprot_file_name):
    words_list = []
    pattern = re.compile('EC (?:\d{1,2}|-).(?:\d{1,2}|-).(?:\d{1,2}|-).(?:\d{1,2}|-)')
    with open(uniprot_file_name, "r") as file:
        for line in file:
            skip_line = False
            # if line.startswith("Entry"): skip_line = True
            if not skip_line:
                words = line.split('(')
                for word in words:
                    word = word.replace(')', '')
                    word = word.replace('\n', '')
                    words_line = word.split(('\t'))
                    if len(words_line) > 1:
                        for w in words_line:
                            w = w.strip()
                            if w not in words_list:
                                if pattern.search(w) is None:
                                    words_list.append(w)
                    elif len(words_line) == 1:
                        w2 = words_line[0].strip()
                        if pattern.search(w2) is None:
                            words_list.append(w2)
    return words_list


def get_tokens_protein_names_n_ids(uniprot_file_name):
    prot_list = get_protein_names_n_ids(uniprot_file_name)
    tokens_list = []
    pattern = re.compile('[0-9]+')
    pattern2 = re.compile('[/,!.?]+')
    for full_name in prot_list:
        tokens = wordpunct_tokenize(full_name)
        for token in tokens:
            if token != '-' and token != ',' and token != '--' and len(token) > 3:
                if pattern.search(token) is None and pattern2.search(token) is None:
                    if token not in tokens_list:
                        tokens_list.append(token)
    return tokens_list


def serialize_features(features_dataframe, path = '../tests/features/features.txt'):
    with open(path, 'wb') as file:
        pickle.dump(features_dataframe, file)


def deserialize_features(path):
    with open(path, 'rb') as file:
        features_dataframe = pickle.load(file)
    return features_dataframe


method_by_feature = { 'nmr_tokens': nmr_tokens,
                      'sentence_size': sentence_size_stats,
                      'title_size': title_sentence_size,
                      'vowels': nmr_vowels,
                      'stems': distinct_stems,
                      'unusualwords': nmr_unusual_words,
                      'termlists': mutation_strong_weak,
                      'termlists_2': mutation_interaction_impact,
                      'termlists_3': mutation_interaction_impact_v2,
                      'termlists_4': mutation_interaction_impact_sentences,
                      'termlists_5': mutation_interaction_impact_sentences_v2,
                      'termlists_6': mutation_interaction_impact_title,
                      'protein': nmr_protein_names_n_ids,
                      'protein_2': nmr_protein_tokens,
                      'protein_3': nmr_protein_tokens_v2,
                      'protein_4': nmr_protein_tokens_lower,
                      'summary': mutation_interaction_impact_summary,
                      'summary_2': mutation_interaction_impact_summary_v2,
                      'summary_3': strong_weak_mutation_summary,
                      'stopwords': nmr_stopwords,
                      'textsize': fulltext_size,
                      'tokens_size': tokens_size_stats,
                      'capital': capital_letters,
                      'punctuation': punctuation,
                      'POS': POS_counter,
                      'prot_terms': mut_prot_sent,
                      'tfidf': tfidf,
                    }





# if __name__== "__main__":
#     #protein_names_list = get_tokens_protein_names_n_ids('term_lists/uniprot.tab')
#     #print(protein_names_list[0:20])
#     # stopWords = set(stopwords.words('english'))
#     # print(stopWords)
#     # count = lambda l1, l2: sum([1 for x in l1 if x in l2])
#     # s = ' wiqod. qwodiqwd!'
#     # print(count(s, set(string.punctuation)))
#     #print(strong_mutation_list)
