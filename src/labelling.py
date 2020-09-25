import numpy as np

def labelling(n_examples, n_classes):
    repetition = int(n_examples / n_classes)+1

    label = np.ones((n_examples, ), dtype=int)
    for i in range(n_classes):
        label[i * repetition : (i + 1) * repetition] = i

    return label