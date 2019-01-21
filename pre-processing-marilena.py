from music21 import converter, instrument, note, chord, stream
import numpy as np
import pickle
import glob
from keras.utils import np_utils
import os

w = 15 # dimensione finestra
max_quarter = 8 # numero massimo di quarti di battuta
epochs = 500
batch_size = 128
m = 32 # dimensione embedding note/durate
k = 32 # dimensione embedding note e durate insieme
training = False
skip_load = False
model_path = 'models'
model_name = 'all2model_midisongs_n' + str(w) + '_m' + str(m) + '_k' + str(k) + '_epochs' + str(epochs) + '_batchsize' + str(batch_size) + '_maxquarter' + str(max_quarter)
model_file = model_path + '/' + model_name

def get_data(max_quarter, skip_load):
    dataset_path = 'midi_songs'
    notes = []
    durations = []



    for file in glob.glob(dataset_path+'/*.mid'):
        midi = converter.parse(file)

        print('Parsing {}'.format(file))

        notes_to_parse = []
        try:  # il file ha le parti dello strumento
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except:  # il file ha le note in una struttura schiacciata
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):  # se è una nota
                s = int(4*float(element.duration.quarterLength))

                if s > 0:
                    durations.append(s if s < max_quarter else max_quarter-1)

                    n = str(element.pitch)
                    notes.append(n)
            elif isinstance(element, chord.Chord):  # se è un accordo
                s = int(4*float(element.duration.quarterLength))

                if s > 0:
                    durations.append(s if s < max_quarter else max_quarter-1)

                    n = '.'.join(str(n) for n in element.normalOrder)
                    notes.append(n)

    durations = np.asarray(durations).astype('int32')

    with open("data"'/seen_notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    with open("data"+'/seen_durations', 'wb') as filepath:
        pickle.dump(durations, filepath)

    return notes, durations


def get_sequences(notes, durations, w):
    # prendo i nomi dei pitch
    pitchnames = sorted(set(item for item in notes))

    # creo un dizionario per mappare pitch a interi (valori corrispondenti alle note)
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    duration_input = []
    duration_output = []
    note_input = []
    note_output = []

    # creo sequenze di input e output
    for i in range(0, len(notes) - w, 1):
        sequence_in = notes[i:i + w]
        sequence_out = notes[i + w]
        seq = [note_to_int[char] for char in sequence_in]
        note_input.append(seq)
        note_output.append(note_to_int[sequence_out])

        duration_input.append(durations[i:i + w])
        duration_output.append(durations[i + w])

    n_patterns = len(note_input)

    note_input = np.reshape(note_input, (n_patterns, w, 1))
    # normalizzo l'input
    # note_input = note_input / float(n_vocab)

    note_output = np.asarray(note_output)
    note_input = np_utils.to_categorical(note_input)

    duration_output = np.asarray(duration_output)
    duration_input = np_utils.to_categorical(duration_input)

    return note_input, note_output, duration_input, duration_output


def one_hot_encode(x, c):
    oh = np.zeros((x.shape[0], c))
    oh[np.arange(x.shape[0]), x] = 1
    return oh


def one_hot_decode(x):
    y = np.argmax(x, axis=1).astype('float64')
    return y

def main():
    global epochs, batch_size, w, max_quarter, m, k, training, model_file
    notes, durations = get_data(max_quarter, skip_load)

    n=notes
    d=durations

    note_input, note_output, duration_input, duration_output=get_sequences(notes, durations, w)
    ni=note_input
    no=note_output
    di=duration_input
    do=duration_output



if __name__ == '__main__':
    main()