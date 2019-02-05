import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
from matplotlib import pyplot as plt

def prediction(model, note_input, duration_input, note_output, duration_output, vocab, n_vocab, time_vocab):
    p_1, p_2 = model.predict(x=[note_input, duration_input])

    int_to_note = dict((number, note) for number, note in enumerate(vocab))

    p1 = np.asarray(p_1)
    p2 = np.asarray(p_2)

    p1 = np.argmax(p1, axis=1)  # tutte le note predette (14117, 354)
    p2 = np.argmax(p2, axis=1)  # tutte le durate predette (14117, 8)


    note_cm = matrice_confusione(p_1, note_output, n_vocab)

    time_cm = matrice_confusione(p_2, duration_output, time_vocab)

    print(note_cm)
    print(time_cm)

    pnotes = []
    for i in range(len(p_1)):
        pnote = int_to_note[p1[i]]  # nota associata al valore numerico della nota
        pnotes.append(pnote)


    return pnotes, p_2

def original_prediction(vocab, note_y_test):
    int_to_note = dict((number, note) for number, note in enumerate(vocab))
    nnote_y_test = np.asarray(note_y_test)
    pnotes = []
    for i in range(len(nnote_y_test)):
        pnote = int_to_note[nnote_y_test[i]]  # nota associata al valore numerico della nota
        pnotes.append(pnote)

    return pnotes

def matrice_confusione(predetti, originali, n_vocab):
    m = np.zeros(shape=(n_vocab,n_vocab), dtype=float)
    for i in range(len(predetti)):
        p = np.unravel_index(np.argmax(predetti[i]), predetti[i].shape)
        o = np.unravel_index(np.argmax(originali[i]), originali[i].shape)
        m[o[0]][p[0]] += 1

    for i in range(len(m)):
        somma = np.sum(m[i])
        m[i] /= somma
    m = m.round(4)
    m = m*100

    return m