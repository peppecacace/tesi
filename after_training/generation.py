import numpy as np

def free_run(model, note_input, duration_input, n_vocab, max_quarter, int_to_note):


    '''seed_n = np.random.randint(1, n_vocab, size=[w])  # seed1 - RANDOM
    seed_t = np.random.randint(1, max_quarter, size=[w])  # seed2

    random_notes = []
    for i in range(len(seed_n)):
        random_notes.append(int_to_note[seed_n[i]])

    seed_n = np_utils.to_categorical(seed_n, n_vocab)
    seed_t = np_utils.to_categorical(seed_t, max_quarter)

    pattern_note = np.reshape(seed_n, (1, w, n_vocab))
    pattern_time = np.reshape(seed_t, (1, w, max_quarter))

    random_times = pattern_time
    random_times = np.reshape(random_times, (w, 1, max_quarter))'''

    seed = np.random.randint(0, len(note_input) - 1)  #RANDOM TRA TRAINING E TEST

    #seed = np.random.randint(0, len(note_input[dslice:]) - 1)  #RANDOM TRA TEST SET

    #seed
    pattern_note = note_input[seed]  # finestra random del dataset
    pattern_time = duration_input[seed]

    pattern_note = np.reshape(pattern_note, (1, note_input.shape[1], n_vocab))
    pattern_time = np.reshape(pattern_time, (1, duration_input.shape[1], max_quarter))

    #original
    original_notes = note_input[seed+1:seed+4]
    original_times = duration_input[seed+1:seed+4]

    original_notes = np.reshape(original_notes, (1, note_input.shape[1]*3, n_vocab))
    original_times = np.reshape(original_times, (1, duration_input.shape[1]*3, max_quarter))
    original_times = np.asarray(original_times)
    originaltimes = np.reshape(original_times, (original_times.shape[1], original_times.shape[0], original_times.shape[2]))

    original_notes = np.argmax(original_notes, axis=-1)

    originalnotes = []
    for i in range(original_notes.shape[-1]):
        el = int_to_note[int(original_notes[:, i])]
        originalnotes.append(el)

    originalnotes = np.asarray(originalnotes)

    #randomgen
    random_times = pattern_time
    random_times = np.asarray(random_times)
    randomtimes = np.reshape(random_times, (random_times.shape[1], random_times.shape[0], random_times.shape[2]))
    random_notes = pattern_note
    random_notes = np.argmax(random_notes, axis=-1)


    randomnotes = []
    for i in range(random_notes.shape[-1]):
        elem = int_to_note[int(random_notes[:, i])]
        randomnotes.append(elem)

    randomnotes = np.asarray(randomnotes)

    prediction_notes = []
    prediction_times = []

    for next_index in range(100):
        prediction_note, prediction_time = model.predict([pattern_note, pattern_time])

        indexnote = np.argmax(prediction_note)
        resultnote = int_to_note[indexnote]  # nota associata al valore numerico della nota

        prediction_notes.append(resultnote)
        prediction_times.append(prediction_time)

        prediction_note = np.reshape(prediction_note, (1, 1, prediction_note.shape[1]))
        pattern_note = np.concatenate((pattern_note[:, 1:, :], prediction_note), axis=1)

        prediction_time = np.reshape(prediction_time, (1, 1, prediction_time.shape[1]))
        pattern_time = np.concatenate((pattern_time[:, 1:, :], prediction_time), axis=1)


    prediction_notes = np.asarray(prediction_notes)
    prediction_times = np.asarray(prediction_times)

    return prediction_notes, prediction_times, randomnotes, randomtimes, originalnotes, originaltimes