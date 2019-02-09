import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
from keras.utils import np_utils
import os
import random

def get_notes(max_quarter):
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    dataset_path = 'midi_songs'
    skip_path = 'processing/data'

    allnotes = []
    alldurations = []
    songs = []

    if os.path.isfile(skip_path+'/seen_notes') and os.path.isfile(skip_path+'/seen_durations') and os.path.isfile(skip_path+'/seen_songs'):

        allnotes = None
        alldurations = None
        songs = None

        with open(r''+skip_path+'/seen_notes', "rb") as input_file:
            allnotes = pickle.load(input_file)

        with open(r''+skip_path+'/seen_durations', "rb") as input_file:
            alldurations = pickle.load(input_file)

        with open(r''+skip_path+'/seen_songs', "rb") as input_file:
            songs = pickle.load(input_file)

        return allnotes, alldurations, songs

    for file in glob.glob(dataset_path + '/*.mid'):
        midi = converter.parse(file)

        notes = []
        durations = []

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
                    alldurations.append(s if s < max_quarter else max_quarter - 1)

                    n = str(element.pitch)
                    notes.append(n)
                    allnotes.append(n)
            elif isinstance(element, chord.Chord):  # se è un accordo
                s = int(4 * float(element.duration.quarterLength))

                if s > 0:
                    durations.append(s if s < max_quarter else max_quarter - 1)
                    alldurations.append(s if s < max_quarter else max_quarter - 1)

                    n = '.'.join(str(n) for n in element.normalOrder)
                    notes.append(n)
                    allnotes.append(n)

        songs.append([notes, durations])

    random.shuffle(songs)

    with open("processing/data" + '/seen_notes', 'wb') as filepath:
        pickle.dump(allnotes, filepath)

    with open("processing/data" + '/seen_durations', 'wb') as filepath:
        pickle.dump(alldurations, filepath)

    with open("processing/data" + '/seen_songs', 'wb') as filepath:
        pickle.dump(songs, filepath)

    return allnotes, alldurations, songs

def prepare_sequences(allnotes, w, songs, note_to_int):
    """ Prepare the sequences used by the Neural Network """

    note_input = []
    note_output = []

    duration_input = []
    duration_output = []

    # create input sequences and the corresponding outputs

    for song in songs:
        notes = song[0]
        durations = song[1]

        for i in range(0, len(notes) - w, 1):     ##for (inizio,fine,passo)
            sequence_in = []
            for j in range(w):
                sequence_in.append(note_to_int[notes[i+j]])
            note_input.append(sequence_in)
            note_output.append(note_to_int[notes[i + w]])

            duration_input.append(durations[i:i + w])
            duration_output.append(durations[i + w])

    n_patterns = len(note_input)


    note_output = np_utils.to_categorical(note_output)
    note_input = np_utils.to_categorical(note_input)

    duration_output = np_utils.to_categorical(duration_output)
    duration_output = numpy.delete(duration_output, 0, axis=1)

    duration_input = np_utils.to_categorical(duration_input)
    duration_input = numpy.delete(duration_input, 0 , axis=2)

    return note_input, note_output, duration_input, duration_output
