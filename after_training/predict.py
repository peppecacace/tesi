import numpy as np
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

def prediction(model, note_input, duration_input, note_output, duration_output, n_vocab, time_vocab, int_to_note):
    p_1, p_2 = model.predict(x=[note_input, duration_input])

    p1 = np.asarray(p_1)
    p2 = np.asarray(p_2)

    p11 = np.asarray(note_output)
    p_11 = np.argmax(p11, axis=1)

    p1 = np.argmax(p1, axis=1)  # tutte le note predette
    #p2 = np.argmax(p2, axis=1)  # tutte le durate predette

    cnotepredette = []
    cnoteoriginali = []


    pnotes = []
    p2notes = []
    for i in range(len(p1)):
        pnote = int_to_note[p1[i]]  # nota associata al valore numerico della nota
        pnotes.append(pnote)

    for i in range(len(p_11)):
        pnote = int_to_note[p_11[i]]
        p2notes.append(pnote)

    for i in range(len(p2notes)):
        if "." not in str(p2notes[i]):
            if str.isnumeric(p2notes[i]) == False:
                if "." not in str(pnotes[i]):
                    if str.isnumeric(pnotes[i]) == False:
                        cnoteoriginali.append(p2notes[i])
                        cnotepredette.append(pnotes[i])



    cm = confusion_matrix(p2notes, pnotes)
    n_classes = cm.shape[0]
    with open('outfile.txt', 'wb') as f:
        for line in cm:
            np.savetxt(f, line, fmt='%.2f')
    print(cm)

    return pnotes, p_2, cm, n_classes


def plot_confusion_matrix(conf_mat,
                          hide_spines=False,
                          hide_ticks=False,
                          figsize=None,
                          cmap=None,
                          colorbar=False,
                          show_absolute=True,
                          show_normed=False):

    if not (show_absolute or show_normed):
        raise AssertionError('Both show_absolute and show_normed are False')

    total_samples = conf_mat.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    if cmap is None:
        cmap = plt.cm.Blues

    if figsize is None:
        figsize = (len(conf_mat) * 1.25, len(conf_mat) * 1.25)

    if show_absolute:
        matshow = ax.matshow(conf_mat, cmap=cmap)
    else:
        matshow = ax.matshow(total_samples, cmap=cmap)

    if colorbar:
        fig.colorbar(matshow)

    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            cell_text = ""
            if show_absolute:
                cell_text += format(conf_mat[i, j], 'd')
                if show_normed:
                    cell_text += "\n" + '('
                    cell_text += format(conf_mat[i, j], '.2f') + ')'
            else:
                cell_text += format(conf_mat[i, j], '.2f')
            ax.text(x=j,
                    y=i,
                    s=cell_text,
                    va='center',
                    ha='center',
                    color="white" if conf_mat[i, j] > 0.5 else "black")

    if hide_spines:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    if hide_ticks:
        ax.axes.get_yaxis().set_ticks([])
        ax.axes.get_xaxis().set_ticks([])

    plt.xlabel('predicted label')
    plt.ylabel('true label')
    return fig, ax
