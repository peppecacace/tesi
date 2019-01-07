import glob
import pickle
import numpy
import matplotlib
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot

def train_network():
    """ Train a Neural Network to generate music """
    notes = get_notes()

    # get amount of pitch names
    n_vocab = len(set(notes))

    network_input, network_output = prepare_sequences(notes, n_vocab)

    model = create_network(network_input, n_vocab)

    t = train(model, network_input, network_output)

    #plottraining(t)

def get_notes():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []

    for file in glob.glob("midi_songs/*.mid"):
        midi = converter.parse(file)            ##questo cerca di analizzare ciò che gli passo in un flusso

        print("Parsing %s" % file)

        notes_to_parse = None

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)     ##partiziona in base allo strumento e dopo prende solo una parte(?)
            notes_to_parse = s2.parts[0].recurse()
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:              ##ora controllo se sono note o accordi
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes

def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 15

    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

     # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):     ##for (inizio,fine,passo)
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input

    network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)

    return (network_input, network_output)

def create_network(network_input, n_vocab):
    """ create the structure of the neural network """
    model = Sequential()
    model.add(LSTM(
        units=256,
        input_shape=(network_input.shape[1:]),
        return_sequences=True
    ))                      ##LSTM(unità, dimensionalità degli ingressi,se mandare in uscita tutta la sequenza o una parte(?))
    #model.add(Dropout(0.3))     ##dropout =  Fraction of the units to drop for the linear transformation of the inputs.
    model.add(LSTM(units=256, return_sequences=True))
    #model.add(Dropout(0.3))
    #model.add(LSTM(256))
    #model.add(Dense(256))       ##a quanto pare ti genera una rete "normale"
    #model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    ##To calculate the loss for each iteration of the training we
    ## will be using categorical cross entropy since each of our outputs
    ## only belongs to a single class and we have more than two classes to work with.
    ## And to optimise our network we will use a RMSprop optimizer as it is usually a very good choice for recurrent neural networks.

    return model

def train(model, network_input, network_output):
    """ train the neural network """
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='accuracy',
        verbose=0,
        save_best_only=True,
        mode='max'
    )
    callbacks_list = [checkpoint]
    epochs = 10
    batch_size = 64


    t = model.fit(network_input, network_output, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list,validation_data=(network_input,network_output))

    return t

def plottraining(t):


#da vedere meglio
    accuracy = t.history['acc']
    val_accuracy = t.history['val_acc']
    loss = t.history['loss']
    val_loss = t.history['val_loss']
    epochs = range(len(accuracy))
    pyplot.plot(epochs, accuracy, 'bo', label='Training accuracy')
    pyplot.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    pyplot.title('Training and validation accuracy')
    pyplot.legend()
    pyplot.figure()
    pyplot.plot(epochs, loss, 'bo', label='Training loss')
    pyplot.plot(epochs, val_loss, 'b', label='Validation loss')
    pyplot.title('Training and validation loss')
    pyplot.legend()
    pyplot.show()



if __name__ == '__main__':
    train_network()