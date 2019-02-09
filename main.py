import numpy as np
from processing import pre_processing, training
from after_training import generation, predict
import music21

w = 15 #la nostra sequenza
max_quarter = 8   #massimi quarti di battuta
epochs = 100
batch_size = 32
train = True

def main():
    allnotes, alldurations, songs = pre_processing.get_notes(max_quarter)

    vocab = sorted(
    set(item for item in allnotes))  # nel set trovo l'alfabeto di note viste (associate al pitch) - pitch names, mi serve per la generazione dei midi
    print('pitch names: ' + str(vocab))

    n_vocab = len(set(allnotes))
    print('pitch length: ' + str(n_vocab))

    durations_vocab = len(set(alldurations))
    print('durations length: ' + str(durations_vocab))

        # dizionario sia da note ad interi che viceversa
    note_to_int = dict((note, number) for number, note in enumerate(vocab))

    int_to_note = dict((number, note) for number, note in enumerate(vocab))


    if(train):

        note_input, note_output, duration_input, duration_output = pre_processing.prepare_sequences(allnotes, w, songs, note_to_int)

        model = training.create_network(note_input, n_vocab, duration_input, durations_vocab)


    n = note_input.shape[0]
    print('dimensioni dataset: ' + str(note_input.shape[0]) + ' finestre da ' + str(
        note_input.shape[1]) + ' con dimensione one-hot di ' + str(note_input.shape[2]))

    dslice = int(n * 1 / 10)

    note_x_train = note_input[:dslice*8]
    note_x_val = note_input[dslice*8:dslice*9]
    note_x_test = note_input[dslice*9:]

    time_x_train = duration_input[:dslice*8]
    time_x_val = duration_input[dslice*8:dslice*9]
    time_x_test = duration_input[dslice*9:]

    note_y_train = note_output[:dslice*8]
    note_y_val = note_output[dslice*8:dslice*9]
    note_y_test = note_output[dslice*9:]

    time_y_train = duration_output[:dslice*8]
    time_y_val = duration_output[dslice*8:dslice*9]
    time_y_test = duration_output[dslice*9:]

    #t = training.train(model, note_x_train, note_y_train, time_x_train, time_y_train, epochs, batch_size, note_x_val, note_y_val, time_x_val, time_y_val)

    #training.plottraining(t)
    model.load_weights('mymodel')

    scores2 = model.evaluate([note_x_test, time_x_test], [note_y_test, time_y_test])


    print(
        'valutazione sul test set : output_notes_loss: {} - output_times_loss: {} - output_notes_acc: {} - output_times_acc: {}'.format(scores2[1],
                                                                                                             scores2[2],
                                                                                                             scores2[3],
                                                                                                             scores2[
                                                                                                                 4]))

    # GENERAZIONE
    generation_note, generation_time, seed_note, seed_time, original_note, original_time = generation.free_run(model, note_input,
                                                                                                    duration_input,
                                                                                                     n_vocab,
                                                                                                    durations_vocab, int_to_note)
    create_midi(generation_note, generation_time,
                'generation')  # generation note -> note argmaxate - generation_time -> durate non argmaxate

    create_midi(original_note, original_time, 'generationoriginal')

    create_midi(seed_note, seed_time, 'generationseed')

    # PREDIZIONE
    p_note, p_time = predict.prediction(model, note_x_test, time_x_test, note_y_test, time_y_test, n_vocab, durations_vocab, int_to_note)
    create_midi(p_note, p_time, 'prediction')

    # ORIGINALE PREDIZIONE, per vedere di quanto è diverso l'originale dalla predizione fatta, da errori di esecuzioni da vedere meglio
    pnotes = predict.original_prediction(vocab, note_y_test, int_to_note)
    create_midi(pnotes, time_y_test, 'original')


def create_midi(note, time, name):
    offset = 0
    output_notes = []

    prediction_time = np.asarray(time)

    if name == 'generation':
        prediction_time = np.argmax(prediction_time, axis=2).astype('float32')
    if name == 'prediction':
        prediction_time = np.argmax(prediction_time, axis=1).astype('float32')
    if name == 'generationoriginal':
        prediction_time = np.argmax(prediction_time, axis=2).astype('float32')
    if name == 'generationseed':
        prediction_time = np.argmax(prediction_time, axis=2).astype('float32')

    # crea oggetti di note e accordi sulla base dei valori generati dal modello
    i = 0
    for pattern in note:

        # pattern è un accordo
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = music21.note.Note(int(current_note))
                new_note.storedInstrument = music21.instrument.Piano()
                notes.append(new_note)
            new_chord = music21.chord.Chord(notes)
            t = prediction_time[i]/4.
            new_chord.quarterLength = float(t)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = music21.note.Note(pattern)
            t = prediction_time[i]/4.
            new_note.quarterLength = float(t)
            new_note.offset = offset
            new_note.storedInstrument = music21.instrument.Piano()
            output_notes.append(new_note)

        i = i + 1

        # increase offset each iteration so that notes do not stack
        offset += 0.5

    midi_stream = music21.stream.Stream(output_notes)

    midi_stream.write('midi', fp='results/' + name + '.mid')

if __name__ == '__main__':
    main()