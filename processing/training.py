from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Input, concatenate, Dropout, Bidirectional
from keras.models import Model
from matplotlib import pyplot as plt


def create_network(note_input,  n_vocab, durations_input, durations_vocab):
    """ create the structure of the neural network """

    input1 = Input((note_input.shape[1], note_input.shape[2]))
    input2 = Input((durations_input.shape[1], durations_input.shape[2]))

    input = concatenate([input1, input2])

    y = LSTM(units=25, return_sequences=False)(input)

    out1 = Dense(units=n_vocab, activation='softmax')(y)
    out2 = Dense(units=durations_vocab, activation='softmax')(y)

    model = Model(inputs=[input1, input2], outputs=[out1, out2])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model

def train(model, note_input, note_output,duration_input,duration_output, epochs, batch_size, note_x_test, note_y_test, time_x_test, time_y_test):

    t = model.fit([note_input, duration_input], [note_output, duration_output], epochs=epochs, batch_size=batch_size, validation_data=([note_x_test, time_x_test], [note_y_test, time_y_test]))
    model.save("mymodel")
    return t

def plottraining(t):
    print(t.history.keys())

# summarize history for accuracy
    plt.plot(t.history['dense_1_acc'])
    plt.plot(t.history['val_dense_1_acc'])
    plt.title('model accuracy notes')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
# summarize history for loss
    plt.plot(t.history['dense_1_loss'])
    plt.plot(t.history['val_dense_1_loss'])
    plt.title('model loss notes')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for accuracy
    plt.plot(t.history['dense_2_acc'])
    plt.plot(t.history['val_dense_2_acc'])
    plt.title('model accuracy durations')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(t.history['dense_2_loss'])
    plt.plot(t.history['val_dense_2_loss'])
    plt.title('model loss durations')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
