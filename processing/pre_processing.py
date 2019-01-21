import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
from keras.utils import np_utils

def get_notes(max_quarter):
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    dataset_path = 'midi_songs'
    notes = []
    durations = []

    for file in glob.glob(dataset_path + '/*.mid'):
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
                s = int(4 * float(element.duration.quarterLength))

                if s > 0:
                    durations.append(s if s < max_quarter else max_quarter - 1)

                    n = str(element.pitch)
                    notes.append(n)
            elif isinstance(element, chord.Chord):  # se è un accordo
                s = int(4 * float(element.duration.quarterLength))

                if s > 0:
                    durations.append(s if s < max_quarter else max_quarter - 1)

                    n = '.'.join(str(n) for n in element.normalOrder)
                    notes.append(n)

    durations = numpy.asarray(durations).astype('int32')

    with open("data" + '/seen_notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    with open("data" + '/seen_durations', 'wb') as filepath:
        pickle.dump(durations, filepath)

    return notes, durations

def prepare_sequences(notes, n_vocab, durations, w):
    """ Prepare the sequences used by the Neural Network """

    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

     # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    note_input = []
    note_output = []

    duration_input = []
    duration_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - w, 1):     ##for (inizio,fine,passo)
        sequence_in = notes[i:i + w]
        sequence_out = notes[i + w]
        note_input.append([note_to_int[char] for char in sequence_in])
        note_output.append(note_to_int[sequence_out])

        duration_input.append(durations[i:i + w])
        duration_output.append(durations[i + w])

    n_patterns = len(note_input)

    note_output = np_utils.to_categorical(note_output)
    note_input = np_utils.to_categorical(note_input)

    # reshape the input into a format compatible with LSTM layers
    # note_input = numpy.reshape(note_input, (n_patterns, w, 1))
    # normalize input

    #note_input = note_input / float(n_vocab)

    duration_output = np_utils.to_categorical(duration_output)
    duration_input = np_utils.to_categorical(duration_input)

    return note_input, note_output, duration_input, duration_output

if __name__ == '__main__':
    max_quarter = 8  # numero massimo di quarti di battuta
    w = 15  # finestra
   # notes, durations = get_notes(max_quarter)

 #    n_vocab = len(set(notes))

 #   note_input, note_output, duration_input, duration_output = prepare_sequences(notes, n_vocab, durations, w)