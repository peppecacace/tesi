from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Input, concatenate
from keras.models import Model
from matplotlib import pyplot as plt
from keras.callbacks import ModelCheckpoint

def create_network(note_input,  n_vocab, durations_input, durations_vocab):
    """ create the structure of the neural network """

    input1 = Input((note_input.shape[1], note_input.shape[2]))
    input2 = Input((durations_input.shape[1], durations_input.shape[2]))

    input = concatenate([input1, input2])

    y = LSTM(units=150, return_sequences=False)(input)

    out1 = Dense(units=n_vocab, activation='softmax')(y)
    out2 = Dense(units=durations_vocab, activation='softmax')(y)

    model = Model(inputs=[input1, input2], outputs=[out1, out2])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model

def train(model, note_input, note_output,duration_input,duration_output, epochs, batch_size, note_x_test, note_y_test, time_x_test, time_y_test):
    checkpoint = ModelCheckpoint(
        filepath="processing/models/model{epoch}.hdf5",
        monitor='loss',
        verbose=0,
        save_best_only=False,
        mode='min'
    )
    callbacks_list = [checkpoint]
    t = model.fit([note_input, duration_input], [note_output, duration_output], epochs=epochs, batch_size=batch_size, validation_data=([note_x_test, time_x_test], [note_y_test, time_y_test]), callbacks=callbacks_list)
    model.save("mymodel")
    return t

def evaluatingmodels(note_x_test, time_x_test, note_y_test, time_y_test, epochs, model):
    scores = []
    filepath = "processing/models"

    for i in range(1, epochs, 1):
        s = filepath + "/model" +str(i) +".hdf5"
        model.load_weights(filepath=s)
        scores.append(model.evaluate([note_x_test, time_x_test], [note_y_test, time_y_test]))
    return scores

def plottraining(t, scores):
    print(t.history.keys())

    note_test_acc = []
    note_test_loss = []

    time_test_acc = []
    time_test_loss = []

    for i in range(len(scores)):
        s = scores[i]
        note_test_acc.append(s[3])
        note_test_loss.append(s[1])
        time_test_acc.append(s[4])
        time_test_loss.append(s[2])


    #accuracy sulle note
    plt.plot(t.history['dense_1_acc'])
    plt.plot(t.history['val_dense_1_acc'])
    plt.plot(note_test_acc)
    plt.title('model accuracy notes')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val', 'test'], loc='upper left')
    plt.show()

    #loss sulle note
    plt.plot(t.history['dense_1_loss'])
    plt.plot(t.history['val_dense_1_loss'])
    plt.plot(note_test_loss)
    plt.title('model loss notes')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val', 'test'], loc='upper left')
    plt.show()

    #accuracy sulle durate
    plt.plot(t.history['dense_2_acc'])
    plt.plot(t.history['val_dense_2_acc'])
    plt.plot(time_test_acc)
    plt.title('model accuracy durations')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val', 'test'], loc='upper left')
    plt.show()

    #loss sulle durate
    plt.plot(t.history['dense_2_loss'])
    plt.plot(t.history['val_dense_2_loss'])
    plt.plot(time_test_loss)
    plt.title('model loss durations')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val', 'test'], loc='upper left')
    plt.show()




