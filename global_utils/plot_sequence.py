
import matplotlib.pyplot as plt


def plot_sequence(*sequence_label_pairs, use_legend=True, title=''):
    for sequence, label in sequence_label_pairs:
        plt.plot(sequence.reshape(-1), label=label)
    if use_legend:
        plt.legend()
    plt.title(title)
    plt.show()
