#!/usr/bin/env python
import numpy as np
import time
import pickle
from sklearn.preprocessing import LabelEncoder
from keras.layers import Input, LSTM, Dense, Activation, Dropout, concatenate
from keras.models import Model, load_model
import tensorflow as tf
from keras.backend import tensorflow_backend as K
from keras.callbacks import ModelCheckpoint

def standardize_windows(array):
    return (array / array[0]) - 1

def load_data(split_size, cv=False filename='data/total_data.csv', window_length=40, standardize=True):
    '''
    Load all data from `data/total_data.csv`
    File rows: Ticker, Start Date, End Date, Sector, Industry, Pattern, Dollar, Interest, Volatility, Prices (window = [0:40], target = [40])
    Does not include auxillary data
    ---
    IN:
        split_size: float [0,1], train_test_split, usually 80%
        filename: `data/total_data.csv`
        window_length: int, size of lookback period; always 40 for now
        standardize: boolean, standardize each window by its first price point

    OUT:
        x_train, y_train, x_test, x_train: train and test datasets
    '''
    raw_data = open(filename).read().split('\n')
    raw_data.pop()

    data = [row.split(',') for row in raw_data]

    prices = np.array([info[9:] for info in data]).astype(float)
    tickers = np.array([info[0] for info in data]).astype(str)
    date_pairs = np.array([(info[1], info[2]) for info in data]).astype(np.datetime64)

    # Standardize
    if standardize:
        prices = np.apply_along_axis(standardize_windows, 1, prices)

    # Set random seed for reproducability and shuffle data
    np.random.seed(42)
    shuffle = np.random.permutation(prices.shape[0])

    shuffled_prices = prices[shuffle]
    shuffled_tickers = tickers[shuffle]
    shuffled_dates = date_pairs[shuffle]

    # Train-test split
    split = int(round(split_size * prices.shape[0]))
    x_train = shuffled_prices[:split, :-1]
    y_train = shuffled_prices[:split, -1]

    x_test = shuffled_prices[split:, :-1]
    y_test = shuffled_prices[split:, -1]

    if not cv:
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return x_train, y_train, x_test, y_test

def build_model(structure, dropout):
    '''
    Build deep NN with Keras. Architecture consists of 2 LSTM layers (40x40, and 40x80), connected to a dense layer (80x1), with that output being the baseline prediction
    ---
    IN:
        structure: list, size of each layer, [1,40,80,1,5] for this model
        dropout: boolean, dropout for the LSTM layers, set to 0.2
    OUT:
        model: compiled baseline model
    '''
    # Time series stock data (40-day lookback window)
    main_input = Input(shape=(structure[1],structure[0]), dtype='float32', name='main_input')
    print(main_input._keras_shape)

    # First-layer LSTM, 40-to-40 mapping
    lstm_one = LSTM(structure[1], return_sequences=True)(main_input)
    if dropout:
        lstm_one = Dropout(.2)(lstm_one)
    print(lstm_one._keras_shape)

    # Second-layer LSTM, 40-to-80 mapping
    lstm_two = LSTM(structure[2], return_sequences=False)(lstm_one)
    if dropout:
        lstm_two = Dropout(.2)(lstm_two)
    print(lstm_two._keras_shape)

    # Fully connected layer for LSTM, returns 1
    lstm_out = Dense(structure[3], activation='linear')(lstm_two)
    print(lstm_out._keras_shape)

    model = Model(inputs=main_input, outputs=lstm_out)

    start = time.time()
    model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])
    print('> Compilation Time : ', time.time() - start)

    return model


if __name__=='__main__':

    global_start_time = time.time()
    structure = [1,40,80,1]
    batch_size = int(input('Batch Size: '))
    epochs = int(input('Epochs: '))
    #split_size = float(input('Split Size: '))
    split_size=0.8
    cv = bool(input('Cross validate? (True or False)'))
    if cv:
        fold = int(input('Fold (1 or 2): '))
    aux = False
    dropout = True

    print('> Loading data... ')

    x_train, y_train, x_test, y_test = load_data(split_size=split_size)

    if cv:
        if fold == 1:

            x_val = x_train[int(x_train.shape[0]/2):]
            y_val = y_train[int(y_train.shape[0]/2):]

            x_train = x_train[:int(x_train.shape[0]/2)]
            y_train = y_train[:int(y_train.shape[0]/2)]

        else:

            x_val = x_train[:int(x_train.shape[0]/2)]
            y_val = y_train[:int(y_train.shape[0]/2)]

            x_train = x_train[int(x_train.shape[0]/2):]
            y_train = y_train[int(y_train.shape[0]/2):]


        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))

    print('> Data Loaded. Duration : %f . Compiling... ' % (time.time() - global_start_time))

    # Utilize multi-core processing
    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=16)) as sess:
        K.set_session(sess)

        model = build_model(structure=structure, aux=aux, dropout=dropout)

        train_start_time = time.time()

        if cv:

            if fold == 1:
                filepath="models/simple_model_fold_1-{epoch:02d}-{val_loss:.2f}.hdf5"
            else:
                filepath="models/simple_model_fold_2-{epoch:02d}-{val_loss:.2f}.hdf5"

            checkpoint = ModelCheckpoint(filepath, monitor='val_loss')

            history = model.fit(
                x_train,
                y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_val, y_val), callbacks=[checkpoint])

            print('Training duration : ', time.time() - train_start_time)
            print('Total duration : ', time.time() - global_start_time)

            model.save('models/simple_%d_fold_%d.h5' % (epochs, fold))
            with open('models/metrics_simple_%d_fold_%d.pkl' % (epochs, fold), 'wb') as f:
                pickle.dump(
            [history.history['mean_squared_error'], history.history['val_mean_squared_error'], history.history['mean_absolute_error'], history.history['val_mean_absolute_error']], f)

        else:
            filepath="models/simple_model-{epoch:02d}-{val_loss:.2f}.hdf5"

            checkpoint = ModelCheckpoint(filepath, monitor='val_loss')

            history = model.fit(
                x_train,
                y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_test, y_test), callbacks=[checkpoint])

            print('Training duration : ', time.time() - train_start_time)
            print('Total duration : ', time.time() - global_start_time)

            model.save('models/simple_%d.h5' % (epochs, fold))
            with open('models/metrics_simple_%d.pkl' % (epochs, fold), 'wb') as f:
                pickle.dump(
            [history.history['mean_squared_error'], history.history['val_mean_squared_error'], history.history['mean_absolute_error'], history.history['val_mean_absolute_error']], f)
