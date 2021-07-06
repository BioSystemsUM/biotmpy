from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, LSTM, RNN, Bidirectional, Flatten, Activation, \
    RepeatVector, Permute, Multiply, Lambda, Concatenate, BatchNormalization
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.metrics import confusion_matrix
from mlearning.attention import AttentionLayer
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import TimeDistributed, GRU
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import RMSprop, Adagrad
from transformers import TFBertModel
from mlearning.attention_context import AttentionWithContext
from tensorflow.keras.layers import SpatialDropout1D
from transformers import TFBertForSequenceClassification, BertConfig
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import f1_score, recall_score, precision_score


def compile(model, optimizer, lr, dl_config, loss, n_classes):
    if not optimizer:
        optimizer = Adam(lr=lr)

    if not loss:
        if n_classes == 2:
            model.compile(loss='binary_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])
        elif n_classes > 2:
            model.compile(loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])

    else:
        model.compile(loss=loss,
            optimizer=optimizer,
            metrics=['accuracy'])

    dl_config.learning_rate = lr

    return model




def Burns_CNNBiLSTM(embedding_matrix, dl_config, n_classes=2, loss=None, learning_rate=None,
                    optimizer=None, seed_value=None):
    max_sent_len = dl_config.max_sent_len
    vocab_size = embedding_matrix.shape[0]
    embed_dim = embedding_matrix.shape[1]
    if not learning_rate:
        if not dl_config.learning_rate:
            learning_rate = 0.001
        else:
            learning_rate = dl_config.learning_rate


    sequence_input = Input(shape=(max_sent_len,), dtype='int32')
    activations = Embedding(vocab_size, embed_dim,
                            weights=[embedding_matrix], input_length=max_sent_len, trainable=False)(sequence_input)
    activations = Dropout(0.4, seed=seed_value)(activations)
    activations = Conv1D(16, 5, strides=1, activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(1e-4))(activations)
    activations = MaxPooling1D(4)(activations)
    activations = Conv1D(16, 5, strides=1, activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(1e-4))(activations)

    activations = Dropout(0.4, seed=seed_value)(activations)
    activations = Bidirectional(LSTM(64))(activations)
    activations = Dropout(0.4, seed=seed_value)(activations)

    attention = Dense(1, activation='tanh')(activations) 
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(128)(attention)
    attention = Permute([2, 1])(attention)

    doc_representation = Multiply()([activations, attention])
    doc_representation = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(128,))(doc_representation)

    output_layer = Dense(1, activation='sigmoid')(doc_representation)

    model = Model(sequence_input, output_layer)
    if not optimizer:
        optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model = compile(model=model,
                    optimizer=optimizer,
                    lr=learning_rate,
                    dl_config=dl_config,
                    loss=loss,
                    n_classes=n_classes)
    model.summary()
    return model


def Hierarchical_Attention_GRU(embedding_matrix, dl_config, n_classes=2, loss=None, learning_rate=None,
                                   optimizer=None, seed_value=None):
    max_sent_len = dl_config.max_sent_len
    max_nb_sentences = dl_config.max_nb_sentences
    vocab_size = embedding_matrix.shape[0]
    embed_dim = embedding_matrix.shape[1]

    if not learning_rate:
        if not dl_config.learning_rate:
            learning_rate = 0.001
        else:
            learning_rate = dl_config.learning_rate

    embedding_layer = Embedding(vocab_size, embed_dim, weights=[embedding_matrix],
                                input_length=max_sent_len, trainable=False, name='word_embedding')

    # Words level attention model
    word_input = Input(shape=(max_sent_len,), dtype='int32', name='word_input')
    word_sequences = embedding_layer(word_input)
    word_gru = Bidirectional(GRU(50, return_sequences=True), name='word_gru')(word_sequences)
    word_dense = Dense(100, activation='relu', name='word_dense')(word_gru)
    word_att, word_coeffs = AttentionLayer(embed_dim, True, name='word_attention')(word_dense)
    wordEncoder = Model(inputs=word_input, outputs=word_att)

    # Sentence level attention model
    sent_input = Input(shape=(max_nb_sentences, max_sent_len), dtype='int32', name='sent_input')
    sent_encoder = TimeDistributed(wordEncoder, name='sent_linking')(sent_input)
    sent_gru = Bidirectional(GRU(50, return_sequences=True), name='sent_gru')(sent_encoder)
    sent_dense = Dense(100, activation='relu', name='sent_dense')(sent_gru)
    sent_att, sent_coeffs = AttentionLayer(embed_dim, return_coefficients=True, name='sent_attention')(sent_dense)
    sent_drop = Dropout(0.5, name='sent_dropout', seed=seed_value)(sent_att)
    preds = Dense(1, activation='sigmoid', name='output')(sent_drop)

    # Model compile
    model = Model(sent_input, preds)
    model = compile(model=model,
                    optimizer=optimizer,
                    lr=learning_rate,
                    dl_config=dl_config,
                    loss=loss,
                    n_classes=n_classes)
    print(wordEncoder.summary())
    print(model.summary())
    return model


def Hierarchical_Attention_GRU2(embedding_matrix, dl_config, n_classes=2, loss=None, learning_rate=None,
                                   optimizer=None, seed_value=None):
    max_sent_len = dl_config.max_sent_len
    max_nb_sentences = dl_config.max_nb_sentences
    vocab_size = embedding_matrix.shape[0]
    embed_dim = embedding_matrix.shape[1]

    if not learning_rate:
        if not dl_config.learning_rate:
            learning_rate = 0.001
        else:
            learning_rate = dl_config.learning_rate

    embedding_layer = Embedding(vocab_size, embed_dim, weights=[embedding_matrix],
                                input_length=max_sent_len, trainable=False, name='word_embedding')

    # Words level attention model
    word_input = Input(shape=(max_sent_len,), dtype='int32', name='word_input')
    word_sequences = embedding_layer(word_input)
    word_gru = Bidirectional(GRU(50, return_sequences=True), name='word_gru')(word_sequences)
    word_dense = Dense(100, activation='relu', name='word_dense')(word_gru)
    word_att, word_coeffs = AttentionLayer(embed_dim, True, name='word_attention')(word_dense)
    wordEncoder = Model(inputs=word_input, outputs=word_att)

    # Sentence level attention model
    sent_input = Input(shape=(max_nb_sentences, max_sent_len), dtype='int32', name='sent_input')
    sent_encoder = TimeDistributed(wordEncoder, name='sent_linking')(sent_input)
    sent_gru = Bidirectional(GRU(50, return_sequences=True), name='sent_gru')(sent_encoder)
    sent_dense = Dense(100, activation='relu', name='sent_dense')(sent_gru)
    sent_att, sent_coeffs = AttentionLayer(embed_dim, return_coefficients=True, name='sent_attention')(sent_dense)
    sent_drop = Dropout(0.5, name='sent_dropout', seed_value=seed_value)(sent_att)
    preds = Dense(1, activation='sigmoid', name='output')(sent_drop)

    # Model compile
    model = Model(sent_input, preds)
    model = compile(model=model,
                    optimizer=optimizer,
                    lr=learning_rate,
                    dl_config=dl_config,
                    loss=loss,
                    n_classes=n_classes)
    print(wordEncoder.summary())
    print(model.summary())
    return model


def Hierarchical_Attention_LSTM(embedding_matrix, dl_config, n_classes=2, loss=None, learning_rate=None,
                                   optimizer=None, seed_value=None):
    max_sent_len = dl_config.max_sent_len
    max_nb_sentences = dl_config.max_nb_sentences
    vocab_size = embedding_matrix.shape[0]
    embed_dim = embedding_matrix.shape[1]

    if not learning_rate:
        if not dl_config.learning_rate:
            learning_rate = 0.001
        else:
            learning_rate = dl_config.learning_rate

    embedding_layer = Embedding(vocab_size, embed_dim, weights=[embedding_matrix],
                                input_length=max_sent_len, trainable=False, name='word_embedding')

    # Words level attention model
    word_input = Input(shape=(max_sent_len,), dtype='int32', name='word_input')
    word_sequences = embedding_layer(word_input)
    word_gru = Bidirectional(LSTM(50, return_sequences=True), name='word_gru')(word_sequences)
    word_dense = Dense(100, activation='relu', name='word_dense')(word_gru)
    word_att, word_coeffs = AttentionLayer(embed_dim, True, name='word_attention')(word_dense)
    wordEncoder = Model(inputs=word_input, outputs=word_att)

    # Sentence level attention model
    sent_input = Input(shape=(max_nb_sentences, max_sent_len), dtype='int32', name='sent_input')
    sent_encoder = TimeDistributed(wordEncoder, name='sent_linking')(sent_input)
    sent_gru = Bidirectional(LSTM(50, return_sequences=True), name='sent_gru')(sent_encoder)
    sent_gru = Dropout(0.2, seed=seed_value)(sent_gru)
    sent_dense = Dense(100, activation='relu', name='sent_dense')(sent_gru)
    sent_att, sent_coeffs = AttentionLayer(embed_dim, return_coefficients=True, name='sent_attention')(sent_dense)
    sent_drop = Dropout(0.5, name='sent_dropout', seed=seed_value)(sent_att)
    preds = Dense(1, activation='sigmoid', name='output')(sent_drop)

    # Model compile
    model = Model(sent_input, preds)
    model = compile(model=model,
                    optimizer=optimizer,
                    lr=learning_rate,
                    dl_config=dl_config,
                    loss=loss,
                    n_classes=n_classes)
    print(wordEncoder.summary())
    print(model.summary())
    return model


def Hierarchical_Attention_LSTM2(embedding_matrix, dl_config, n_classes=2, loss=None, learning_rate=None,
                                   optimizer=None, seed_value=None):
    max_sent_len = dl_config.max_sent_len
    max_nb_sentences = dl_config.max_nb_sentences
    vocab_size = embedding_matrix.shape[0]
    embed_dim = embedding_matrix.shape[1]

    if not learning_rate:
        if not dl_config.learning_rate:
            learning_rate = 0.001
        else:
            learning_rate = dl_config.learning_rate

    embedding_layer = Embedding(vocab_size, embed_dim, weights=[embedding_matrix],
                                input_length=max_sent_len, trainable=False, name='word_embedding')

    # Words level attention model
    word_input = Input(shape=(max_sent_len,), dtype='int32', name='word_input')
    from tensorflow.keras.layers import GaussianNoise
    word_sequences = embedding_layer(word_input)
    word_gru = Bidirectional(LSTM(50, return_sequences=True), name='word_gru')(word_sequences)
    word_gru = GaussianNoise(0.4)(word_gru)
    word_gru = Dropout(0.5, seed=seed_value)(word_gru)
    word_dense = Dense(100, activation='relu', name='word_dense')(word_gru)
    word_att, word_coeffs = AttentionLayer(embed_dim, True, name='word_attention')(word_dense)
    wordEncoder = Model(inputs=word_input, outputs=word_att)

    # Sentence level attention model
    sent_input = Input(shape=(max_nb_sentences, max_sent_len), dtype='int32', name='sent_input')
    sent_encoder = TimeDistributed(wordEncoder, name='sent_linking')(sent_input)
    sent_gru = Bidirectional(GRU(50, return_sequences=True), name='sent_gru')(sent_encoder)
    sent_gru = Dropout(0.5, seed=seed_value)(sent_gru)
    sent_dense = Dense(100, activation='relu', name='sent_dense')(sent_gru)
    sent_att, sent_coeffs = AttentionLayer(embed_dim, return_coefficients=True, name='sent_attention')(sent_dense)
    sent_drop = Dropout(0.5, name='sent_dropout', seed=seed_value)(sent_att)
    preds = Dense(1, activation='sigmoid', name='output')(sent_drop)

    # Model compile
    model = Model(sent_input, preds)
    model = compile(model=model,
                    optimizer=optimizer,
                    lr=learning_rate,
                    dl_config=dl_config,
                    loss=loss,
                    n_classes=n_classes)
    print(wordEncoder.summary())
    print(model.summary())
    from tensorflow.keras.utils import plot_model
    plot_model(wordEncoder, to_file='model_plot1.png', show_shapes=True, show_layer_names=True)
    plot_model(model, to_file='model_plot2.png', show_shapes=True, show_layer_names=True)
    return model


def Hierarchical_Attention_LSTM3(embedding_matrix, dl_config, n_classes=2, loss=None, learning_rate=None,
                                   optimizer=None, seed_value=None):
    max_sent_len = dl_config.max_sent_len
    max_nb_sentences = dl_config.max_nb_sentences
    vocab_size = embedding_matrix.shape[0]
    embed_dim = embedding_matrix.shape[1]

    if not learning_rate:
        if not dl_config.learning_rate:
            learning_rate = 0.001
        else:
            learning_rate = dl_config.learning_rate


    embedding_layer = Embedding(vocab_size, embed_dim, weights=[embedding_matrix],
                                input_length=max_sent_len, trainable=False, name='word_embedding')

    # Words level attention model
    word_input = Input(shape=(max_sent_len,), dtype='int32', name='word_input')
    from tensorflow.keras.layers import GaussianNoise
    word_sequences = embedding_layer(word_input)
    word_gru = Bidirectional(LSTM(100, return_sequences=True), name='word_gru')(word_sequences)
    # word_gru = GaussianNoise(0.1)(word_gru)
    word_dense = Dense(200, activation='relu', name='word_dense')(word_gru)
    word_att, word_coeffs = AttentionLayer(embed_dim, True, name='word_attention')(word_dense)
    wordEncoder = Model(inputs=word_input, outputs=word_att)

    # Sentence level attention model
    sent_input = Input(shape=(max_nb_sentences, max_sent_len), dtype='int32', name='sent_input')
    sent_encoder = TimeDistributed(wordEncoder, name='sent_linking')(sent_input)
    sent_gru = Bidirectional(LSTM(100, return_sequences=True), name='sent_gru')(sent_encoder)
    # sent_gru = Dropout(0.5, seed=seed_value)(sent_gru)
    sent_dense = TimeDistributed(Dense(200, activation='relu', name='sent_dense'))(sent_gru)
    sent_att, sent_coeffs = AttentionLayer(embed_dim, return_coefficients=True, name='sent_attention')(sent_dense)
    # sent_drop = Dropout(0.5,name='sent_dropout', seed=seed_value)(sent_att)
    preds = Dense(1, activation='sigmoid', name='output')(sent_att)

    # Model compile
    model = Model(sent_input, preds)
    model = compile(model=model,
                    optimizer=optimizer,
                    lr=learning_rate,
                    dl_config=dl_config,
                    loss=loss,
                    n_classes=n_classes)
    print(wordEncoder.summary())
    print(model.summary())
    return model


def Hierarchical_Attention_Context(embedding_matrix, dl_config, n_classes=2, loss=None, learning_rate=None,
                                   optimizer=None, seed_value=None):

    max_sent_len = dl_config.max_sent_len
    max_nb_sentences = dl_config.max_nb_sentences
    vocab_size = embedding_matrix.shape[0]
    embed_dim = embedding_matrix.shape[1]

    if not learning_rate:
        if not dl_config.learning_rate:
            learning_rate = 0.001
        else:
            learning_rate = dl_config.learning_rate

    embedding_layer = Embedding(vocab_size, embed_dim, weights=[embedding_matrix],
                                input_length=max_sent_len, trainable=False, name='word_embedding')
    word_input = Input(shape=(max_sent_len,), dtype='int32')
    word = embedding_layer(word_input)
    word = SpatialDropout1D(0.2, seed=seed_value)(word)
    word = Bidirectional(LSTM(128, return_sequences=True))(word)
    word_out = AttentionWithContext()(word)
    wordEncoder = Model(word_input, word_out)

    sente_input = Input(shape=(max_nb_sentences, max_sent_len), dtype='int32')
    sente = TimeDistributed(wordEncoder)(sente_input)
    sente = SpatialDropout1D(0.2, seed=seed_value)(sente)
    sente = Bidirectional(LSTM(128, return_sequences=True))(sente)
    sente = AttentionWithContext()(sente)
    preds = Dense(1, activation='sigmoid')(sente)
    model = Model(sente_input, preds)
    model = compile(model=model,
                    optimizer=optimizer,
                    lr=learning_rate,
                    dl_config=dl_config,
                    loss=loss,
                    n_classes=n_classes)

    print(wordEncoder.summary())
    print(model.summary())
    return model


def Bert_Dense(dl_config, n_classes=2, loss=None, learning_rate=None,
                optimizer=None, seed_value=None, static_bert=True, bert_name_or_path="bert-base-uncased", bert_config=False):
    max_sent_len = dl_config.max_sent_len

    if not learning_rate:
        if not dl_config.learning_rate:
            learning_rate = 0.001
        else:
            learning_rate = dl_config.learning_rate


    idx = Input((max_sent_len), dtype="int32", name="input_idx")
    masks = Input((max_sent_len), dtype="int32", name="input_masks")
    segments = Input((max_sent_len), dtype="int32", name="input_segments")
    
    ## pre-trained bert
    if bert_config:
        bert_config = BertConfig.from_json_file(bert_name_or_path+ '/bert_config.json')
        bert_model = TFBertModel.from_pretrained(bert_name_or_path, from_pt=True, config = bert_config)
    else:
        bert_model = TFBertModel.from_pretrained(bert_name_or_path)
    embedding = bert_model([idx, masks, segments])[0]

    ## fine-tuning
    x = GlobalAveragePooling1D()(embedding)
    x = Dense(100, activation="relu")(x)


    if n_classes==2:
        y_out = Dense(1, activation='sigmoid')(x)
    elif n_classes>2:
        y_out = Dense(n_classes, activation='softmax')(x)


    model = Model([idx, masks, segments], y_out)

    if static_bert:
        for layer in model.layers[:4]:
            layer.trainable = False


    model = compile(model=model,
                    optimizer=optimizer,
                    lr=learning_rate,
                    dl_config=dl_config,
                    loss=loss,
                    n_classes=n_classes)

    print(model.summary())
    return model

def Bert_LSTM(dl_config, n_classes=2, loss=None, learning_rate=None,
                optimizer=None, seed_value=None, static_bert=True, bert_name_or_path="bert-base-uncased", bert_config=False):
    max_sent_len = dl_config.max_sent_len

    if not learning_rate:
        if not dl_config.learning_rate:
            learning_rate = 0.001
        else:
            learning_rate = dl_config.learning_rate


    idx = Input((max_sent_len), dtype="int32", name="input_idx")
    masks = Input((max_sent_len), dtype="int32", name="input_masks")
    segments = Input((max_sent_len), dtype="int32", name="input_segments")
    
    ## pre-trained bert
    if bert_config:
        bert_config = BertConfig.from_json_file(bert_name_or_path+ '/bert_config.json')
        bert_model = TFBertModel.from_pretrained(bert_name_or_path, from_pt=True, config = bert_config)
    else:
        bert_model = TFBertModel.from_pretrained(bert_name_or_path)
    embedding = bert_model([idx, masks, segments])[0]

    ## fine-tuning
    x = Bidirectional(LSTM(50, return_sequences=True,  recurrent_dropout=0.1))(embedding)
    x = Dropout(0.1, seed=seed_value)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(62, activation="relu")(x)
    x = Dropout(0.2, seed=seed_value)(x)


    if n_classes==2:
        y_out = Dense(1, activation='sigmoid')(x)
    elif n_classes>2:
        y_out = Dense(n_classes, activation='softmax')(x)


    model = Model([idx, masks, segments], y_out)

    if static_bert:
        for layer in model.layers[:4]:
            layer.trainable = False

    model = compile(model=model,
                    optimizer=optimizer,
                    lr=learning_rate,
                    dl_config=dl_config,
                    loss=loss,
                    n_classes=n_classes)

    print(model.summary())
    return model




def Bert_CLS(dl_config, n_classes=2, loss=None, learning_rate=None,
                optimizer=None, seed_value=None, static_bert=True, bert_name_or_path="bert-base-uncased", bert_config=False):
    max_sent_len = dl_config.max_sent_len

    if bert_config:
        bert_config = BertConfig.from_json_file('./' +  bert_name_or_path + '/bert_config.json')
        bert_model = TFBertModel.from_pretrained(bert_name_or_path, from_pt=True, config = bert_config)
    else:
        bert_model = TFBertModel.from_pretrained(bert_name_or_path, config=bert_config)

    idx = Input((max_sent_len), dtype="int32", name="input_idx")
    masks = Input((max_sent_len), dtype="int32", name="input_masks")
    segments = Input((max_sent_len), dtype="int32", name="input_segments")

    embedding_layer = bert_model(idx, attention_mask=masks, token_type_ids=segments)[0]
    cls_token = embedding_layer[:,0,:]
    X = BatchNormalization()(cls_token)
    X = Dense(192, activation='relu')(X)
    X = Dropout(0.2, seed=seed_value)(X)

    if n_classes==2:
        y_out = Dense(1, activation='sigmoid')(X)
    elif n_classes>2:
        y_out = Dense(n_classes, activation='softmax')(X)

    model = Model([idx, masks, segments], y_out)

    if static_bert:
        for layer in model.layers[:4]:
            layer.trainable = False

    model = compile(model=model,
                    optimizer=optimizer,
                    lr=learning_rate,
                    dl_config=dl_config,
                    loss=loss,
                    n_classes=n_classes)

    print(model.summary())
    return model




def Bert_Sequence(dl_config, n_classes=2, loss=None, learning_rate=None,
                         optimizer=None, seed_value=None, static_bert=True, bert_name_or_path="bert-base-uncased", bert_config=False):
    max_sent_len = dl_config.max_sent_len
    idx = Input((max_sent_len), dtype="int32", name="input_idx")
    masks = Input((max_sent_len), dtype="int32", name="input_masks")
    segments = Input((max_sent_len), dtype="int32", name="input_segments")
    ## pre-trained bert


    bert_config = BertConfig(num_labels=n_classes).output_hidden_states=False
    bert_senquence = TFBertForSequenceClassification.from_pretrained(bert_name_or_path, config=bert_config)
    y_out = bert_senquence([idx, masks, segments])[0]

    model = Model([idx, masks, segments], y_out)


    if static_bert:
        for layer in model.layers[:4]:
            layer.trainable = False


    model = compile(model=model,
                    optimizer=optimizer,
                    lr=learning_rate,
                    dl_config=dl_config,
                    loss=loss,
                    n_classes=n_classes)


    print(model.summary())
    return model


def Burns_CNN(embedding_matrix, dl_config, n_classes=2, num_filters=64, weight_decay=1e-4, loss=None,
              learning_rate=None, optimizer=None, seed_value=None):
    max_sent_len = dl_config.max_sent_len
    vocab_size = embedding_matrix.shape[0]
    embed_dim = embedding_matrix.shape[1]

    if not learning_rate:
        if not dl_config.learning_rate:
            learning_rate = 0.001
        else:
            learning_rate = dl_config.learning_rate

    model = Sequential()
    model.add(Embedding(vocab_size, embed_dim,
                        weights=[embedding_matrix], input_length=max_sent_len, trainable=False))
    # model.add(Dropout(0.5, seed=seed_value), seed=seed_value)
    model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.5, seed=seed_value))
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Dense(1, activation='sigmoid'))  # multi-label (k-hot encoding)

    optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model = compile(model=model,
                    optimizer=optimizer,
                    lr=learning_rate,
                    dl_config=dl_config,
                    loss=loss,
                    n_classes=n_classes)
    model.summary()
    return model


def Burns_LSTM(embedding_matrix, dl_config, n_classes=2, loss=None, learning_rate=None,
                optimizer=None, seed_value=None):
    max_sent_len = dl_config.max_sent_len
    vocab_size = embedding_matrix.shape[0]
    embed_dim = embedding_matrix.shape[1]

    if not learning_rate:
        if not dl_config.learning_rate:
            learning_rate = 0.001
        else:
            learning_rate = dl_config.learning_rate

    model = Sequential()
    model.add(Embedding(vocab_size, embed_dim,
                        weights=[embedding_matrix], input_length=max_sent_len, trainable=False))
    model.add(Dropout(0.5, seed=seed_value))
    model.add(LSTM(128))
    model.add(Dropout(0.5, seed=seed_value))
    model.add(Dense(1, activation='sigmoid'))

    model = compile(model=model,
                    optimizer=optimizer,
                    lr=learning_rate,
                    dl_config=dl_config,
                    loss=loss,
                    n_classes=n_classes)
    model.summary()
    return model


def Burns_CNN2(embedding_matrix, dl_config, n_classes=2, loss=None, learning_rate=None,
                optimizer=None, seed_value=None):
    max_sent_len = dl_config.max_sent_len
    vocab_size = embedding_matrix.shape[0]
    embed_dim = embedding_matrix.shape[1]

    if not learning_rate:
        if not dl_config.learning_rate:
            learning_rate = 0.001
        else:
            learning_rate = dl_config.learning_rate

    model = Sequential()
    model.add(Embedding(vocab_size, embed_dim,
                        weights=[embedding_matrix], input_length=max_sent_len, trainable=False))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5, seed=seed_value))
    model.add(Dense(1, activation='sigmoid'))
    model = compile(model=model,
                    optimizer=optimizer,
                    lr=learning_rate,
                    dl_config=dl_config,
                    loss=loss,
                    n_classes=n_classes)
    model.summary()
    return model


def Burns_CNN3(embedding_matrix, dl_config, n_classes=2, loss=None, learning_rate=None,
                optimizer=None, seed_value=None):
    max_sent_len = dl_config.max_sent_len
    vocab_size = embedding_matrix.shape[0]
    embed_dim = embedding_matrix.shape[1]

    if not learning_rate:
        if not dl_config.learning_rate:
            learning_rate = 0.001
        else:
            learning_rate = dl_config.learning_rate


    model = Sequential()
    model.add(Embedding(vocab_size, embed_dim,
                        weights=[embedding_matrix], input_length=max_sent_len, trainable=False))
    model.add(Conv1D(64, 7, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(64, 7, activation='relu', padding='same'))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.5, seed=seed_value))
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4)))
    model.add(Dense(1, activation='sigmoid'))  # multi-label (k-hot encoding)

    model = compile(model=model,
                    optimizer=optimizer,
                    lr=learning_rate,
                    dl_config=dl_config,
                    loss=loss,
                    n_classes=n_classes)
    model.summary()
    return model


def Burns_BiLSTM(embedding_matrix, dl_config, n_classes=2, loss=None, learning_rate=None,
                optimizer=None, seed_value=None):
    max_sent_len = dl_config.max_sent_len
    vocab_size = embedding_matrix.shape[0]
    embed_dim = embedding_matrix.shape[1]

    if not learning_rate:
        if not dl_config.learning_rate:
            learning_rate = 0.001
        else:
            learning_rate = dl_config.learning_rate

    model = Sequential()
    model.add(Embedding(vocab_size, embed_dim,
                        weights=[embedding_matrix], input_length=max_sent_len, trainable=False))
    model.add(Bidirectional(LSTM(128)))
    model.add(Dropout(0.5, seed=seed_value))
    model.add(Dense(1, activation='sigmoid'))
    model = compile(model=model,
                    optimizer=optimizer,
                    lr=learning_rate,
                    dl_config=dl_config,
                    loss=loss,
                    n_classes=n_classes)
    model.summary()


def Chollet_DNN(embedding_matrix, dl_config, n_classes=2, loss=None, learning_rate=None,
                optimizer=None, seed_value=None):
    max_sent_len = dl_config.max_sent_len
    vocab_size = embedding_matrix.shape[0]
    embed_dim = embedding_matrix.shape[1]

    if not learning_rate:
        if not dl_config.learning_rate:
            learning_rate = 0.001
        else:
            learning_rate = dl_config.learning_rate

    model = Sequential()
    model.add(Embedding(vocab_size, embed_dim,
                        weights=[embedding_matrix], input_length=max_sent_len, trainable=False))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model = compile(model=model,
                    optimizer=optimizer,
                    lr=learning_rate,
                    dl_config=dl_config,
                    loss=loss,
                    n_classes=n_classes)
    model.summary()
    return model


def DNN(dl_config, n_classes=2, loss=None, learning_rate=None,
            optimizer=None, seed_value=None):

    if not learning_rate:
        if not dl_config.learning_rate:
            learning_rate = 0.001
        else:
            learning_rate = dl_config.learning_rate
    embedding_dim = dl_config.embedding_dim
    max_sent_len = dl_config.max_sent_len
    model = Sequential()
    model.add(Embedding(10_000, embedding_dim, input_length=max_sent_len))
    model.add(Dropout(0.5, seed=seed_value))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model = compile(model=model,
                    optimizer=optimizer,
                    lr=learning_rate,
                    dl_config=dl_config,
                    loss=loss,
                    n_classes=n_classes)
    model.summary()
    return model


def DeepDTA(embedding_matrix, dl_config, NUM_FILTERS=32, FILTER_LENGTH1=5, FILTER_LENGTH2=10, n_classes=2,
            loss=None, learning_rate=None, optimizer=None, seed_value=None):
    max_sent_len = dl_config.max_sent_len
    vocab_size = embedding_matrix.shape[0]
    embed_dim = embedding_matrix.shape[1]
    title_input = Input(shape=(max_sent_len), dtype='int32')  ### Buralar flagdan gelmeliii
    abstract_input = Input(shape=(max_sent_len,), dtype='int32')

    if not learning_rate:
        if not dl_config.learning_rate:
            learning_rate = 0.001
        else:
            learning_rate = dl_config.learning_rate

    ### SMI_EMB_DINMS  FLAGS GELMELII
    encode_title = Embedding(vocab_size, embed_dim,
                             weights=[embedding_matrix], input_length=max_sent_len, trainable=False)(title_input)
    encode_title = Dropout(0.1, seed=seed_value)(encode_title)
    encode_title = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                          strides=1)(encode_title)
    encode_title = Conv1D(filters=NUM_FILTERS * 2, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                          strides=1)(encode_title)

    # encode_title = Bidirectional(LSTM(64))(encode_title)
    encode_title = Conv1D(filters=NUM_FILTERS * 3, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                          strides=1)(encode_title)
    encode_title = GlobalMaxPooling1D()(encode_title)

    encode_abstract = Embedding(vocab_size, embed_dim,
                                weights=[embedding_matrix], input_length=max_sent_len, trainable=False)(abstract_input)
    encode_abstract = Dropout(0.1, seed=seed_value)(encode_abstract)
    encode_abstract = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH2, activation='relu', padding='valid',
                             strides=1)(encode_abstract)
    encode_abstract = Conv1D(filters=NUM_FILTERS * 2, kernel_size=FILTER_LENGTH2, activation='relu', padding='valid',
                             strides=1)(encode_abstract)

    # encode_abstract = Bidirectional(LSTM(64))(encode_abstract)
    encode_abstract = Conv1D(filters=NUM_FILTERS * 3, kernel_size=FILTER_LENGTH2, activation='relu', padding='valid',
                             strides=1)(encode_abstract)
    encode_abstract = GlobalMaxPooling1D()(encode_abstract)

    from tensorflow import keras
    encode_document = keras.layers.concatenate([encode_title, encode_abstract],
                                               axis=-1)  # merge.Add()([encode_smiles, encode_protein])

    FC1 = Dense(1024, activation='relu')(encode_document)
    FC2 = Dropout(0.5, seed=seed_value)(FC1)
    FC2 = Dense(1024, activation='relu')(FC2)
    FC2 = Dropout(0.5, seed=seed_value)(FC2)
    FC2 = Dense(512, activation='relu')(FC2)

    predictions = Dense(1, kernel_initializer='normal')(
        FC2)

    final_Model = Model(inputs=[title_input, abstract_input], outputs=[predictions])
    final_Model = compile(model=final_Model,
                    optimizer=optimizer,
                    lr=learning_rate,
                    dl_config=dl_config,
                    loss=loss,
                    n_classes=n_classes)

    print(final_Model.summary())

    return final_Model


def HAN_opt(embedding_matrix, dl_config, n_classes=2, loss=None, learning_rate=None,
                               optimizer=None, seed_value=None):


    max_sent_len = dl_config.max_sent_len
    max_nb_sentences = dl_config.max_nb_sentences
    vocab_size = embedding_matrix.shape[0]
    embed_dim = embedding_matrix.shape[1]


    if not learning_rate:
        if not dl_config.learning_rate:
            learning_rate = 0.001
        else:
            learning_rate = dl_config.learning_rate


    embedding_layer = Embedding(vocab_size, embed_dim, weights=[embedding_matrix],
                                input_length=max_sent_len, trainable=False, name='word_embedding')
    
    
    word_input = Input(shape=(max_sent_len,), dtype='int32')
    word = embedding_layer(word_input)
    
    word = SpatialDropout1D(0.4, seed=seed_value)(word)
    word = Bidirectional(LSTM(128, return_sequences=True))(word)
    word_out = AttentionWithContext()(word)
    wordEncoder = Model(word_input, word_out)

    
    sente_input = Input(shape=(max_nb_sentences, max_sent_len), dtype='int32')
    sente = TimeDistributed(wordEncoder)(sente_input)
    sente = SpatialDropout1D(0.1, seed = seed_value)(sente)
    
    sente = Bidirectional(LSTM(256, return_sequences=True))(sente)

    sente = AttentionWithContext()(sente)
    preds = Dense(1, activation='sigmoid')(sente)
    model = Model(sente_input, preds)
        

    model = compile(model=model,
                    optimizer=optimizer,
                    lr=learning_rate,
                    dl_config=dl_config,
                    loss=loss,
                    n_classes=n_classes)

    print(model.summary())
    return model


def Bert_Dense_opt(dl_config, n_classes=2, loss=None, learning_rate=None,
                optimizer=None, seed_value=None, static_bert=False, bert_name_or_path="bert-base-uncased", bert_config=False):
    max_sent_len = dl_config.max_sent_len

    if not learning_rate:
        if not dl_config.learning_rate:
            learning_rate = 0.001
        else:
            learning_rate = dl_config.learning_rate


    idx = Input((max_sent_len), dtype="int32", name="input_idx")
    masks = Input((max_sent_len), dtype="int32", name="input_masks")
    segments = Input((max_sent_len), dtype="int32", name="input_segments")
    
    ## pre-trained bert
    if bert_config:
        bert_config = BertConfig.from_json_file(bert_name_or_path+ '/bert_config.json')
        bert_model = TFBertModel.from_pretrained(bert_name_or_path, from_pt=True, config = bert_config)
    else:
        bert_model = TFBertModel.from_pretrained(bert_name_or_path)
    embedding = bert_model([idx, masks, segments])[0]

    ## fine-tuning
    x = GlobalAveragePooling1D()(embedding)
    x = Dropout(0.3, seed=dl_config.seed_value)(x)
    x = Dense(64, activation="relu")(x)


    if n_classes==2:
        y_out = Dense(1, activation='sigmoid')(x)
    elif n_classes>2:
        y_out = Dense(n_classes, activation='softmax')(x)


    model = Model([idx, masks, segments], y_out)

    
    if static_bert:
        for layer in model.layers[:4]:
            layer.trainable = False

    model = compile(model=model,
                    optimizer=optimizer,
                    lr=learning_rate,
                    dl_config=dl_config,
                    loss=loss,
                    n_classes=n_classes)

    print(model.summary())
    return model


def Bert_LSTM_opt(dl_config, n_classes=2, loss=None, learning_rate=None,
                optimizer=None, seed_value=None, static_bert=True, bert_name_or_path="bert-base-uncased", bert_config=False):
    max_sent_len = dl_config.max_sent_len

    if not learning_rate:
        if not dl_config.learning_rate:
            learning_rate = 0.001
        else:
            learning_rate = dl_config.learning_rate


    idx = Input((max_sent_len), dtype="int32", name="input_idx")
    masks = Input((max_sent_len), dtype="int32", name="input_masks")
    segments = Input((max_sent_len), dtype="int32", name="input_segments")
    
    ## pre-trained bert
    if bert_config:
        bert_config = BertConfig.from_json_file(bert_name_or_path+ '/bert_config.json')
        bert_model = TFBertModel.from_pretrained(bert_name_or_path, from_pt=True, config = bert_config)
    else:
        bert_model = TFBertModel.from_pretrained(bert_name_or_path)
    embedding = bert_model([idx, masks, segments])[0]

    ## fine-tuning
    if static_bert:
        lstm_units = 256
        dropout_units = 0.5
        dense_units = 64
    else:
        lstm_units = 64
        dropout_units = 0.2
        dense_units = 256
    x = Bidirectional(LSTM(lstm_units, return_sequences=True,  recurrent_dropout=0.1))(embedding)
    x = Dropout(dropout_units, seed=seed_value)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(dense_units, activation="relu")(x)

    if not static_bert:
        x = Dropout(0.4, seed=dl_config.seed_value)(x)

    if n_classes==2:
        y_out = Dense(1, activation='sigmoid')(x)
    elif n_classes>2:
        y_out = Dense(n_classes, activation='softmax')(x)


    model = Model([idx, masks, segments], y_out)

    if static_bert:
        for layer in model.layers[:4]:
            layer.trainable = False

    model = compile(model=model,
                    optimizer=optimizer,
                    lr=learning_rate,
                    dl_config=dl_config,
                    loss=loss,
                    n_classes=n_classes)

    print(model.summary())
    return model




def Bert_CLS_opt(dl_config, n_classes=2, loss=None, learning_rate=None,
                optimizer=None, seed_value=None, static_bert=True, bert_name_or_path="bert-base-uncased", bert_config=False):
    max_sent_len = dl_config.max_sent_len

    if bert_config:
        bert_config = BertConfig.from_json_file('./' +  bert_name_or_path + '/bert_config.json')
        bert_model = TFBertModel.from_pretrained(bert_name_or_path, from_pt=True, config = bert_config)
    else:
        bert_model = TFBertModel.from_pretrained(bert_name_or_path, config=bert_config)

    idx = Input((max_sent_len), dtype="int32", name="input_idx")
    masks = Input((max_sent_len), dtype="int32", name="input_masks")
    segments = Input((max_sent_len), dtype="int32", name="input_segments")

    embedding_layer = bert_model(idx, attention_mask=masks, token_type_ids=segments)[0]
    cls_token = embedding_layer[:,0,:]
    if static_bert:
        dense_units = 64
        dropout_units = 0.2
    if not static_bert:
        dense_units = 512
        dropout_units = 0.3
    X = Dense(dense_units, activation='relu')(cls_token)
    X = Dropout(dropout_units, seed=seed_value)(X)

    if n_classes==2:
        y_out = Dense(1, activation='sigmoid')(X)
    elif n_classes>2:
        y_out = Dense(n_classes, activation='softmax')(X)

    model = Model([idx, masks, segments], y_out)

    if static_bert:
        for layer in model.layers[:4]:
            layer.trainable = False

    model = compile(model=model,
                    optimizer=optimizer,
                    lr=learning_rate,
                    dl_config=dl_config,
                    loss=loss,
                    n_classes=n_classes)

    print(model.summary())
    return model

