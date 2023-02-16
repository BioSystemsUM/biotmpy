from keras.optimizers.legacy.adam import Adam


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
