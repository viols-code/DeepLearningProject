import numpy as np
import os
import matplotlib.pyplot as plt


def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    Softmax converts all the values such that their sum lies between [0, 1]
    :param x: scores
    :return: softmax values
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def neg_softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    Here we convert the exponential to 1/exponential.
    :param x: scores
    :return: softmax values
    """
    e_x = np.exp(-(x - np.max(x)))
    return e_x / e_x.sum(axis=0)


def sigmoid(z):
    """
    Compute sigmoid values.
    Sigmoid converts each value in the vector in the range [0, 1]
    :param z: scores
    :return: sigmoid values
    """
    g = 1 / (1 + np.exp(-z))
    return g


def check_file_exists(file_path):
    """
    Check the presence of a file.
    :param file_path: file's path
    :return: True if the path exists, false otherwise
    """
    return os.path.exists(file_path)


def save_images_rationales(train1, train2, train3, tokenizer, text, dir):
    for index in range(50):
        if train1[index][2] in ['non-toxic', 'normal']:
            continue

        plt.figure(figsize=(18, 11))

        plt.subplot(131)
        plt.barh(tokenizer.convert_ids_to_tokens(train1[index][0]), train1[index][1], color='skyblue')
        plt.gca().invert_yaxis()
        plt.title('Mean aggregation')
        plt.xlabel('Attention value')

        plt.subplot(132)
        plt.barh(tokenizer.convert_ids_to_tokens(train2[index][0]), train2[index][1], color='skyblue')
        plt.gca().invert_yaxis()
        plt.title('Lenient (OR) aggregation')
        plt.xlabel('Attention value')

        plt.subplot(133)
        plt.barh(tokenizer.convert_ids_to_tokens(train3[index][0]), train3[index][1], color='skyblue')
        plt.gca().invert_yaxis()
        plt.title('Conservative (AND) aggregation')
        plt.xlabel('Attention value')

        plt.tight_layout()
        # Save plot
        plt.savefig(f'Rationales/{dir}/plot_rationales_{index}{text}.png')
        plt.close()


def save_images_zeros(attentions1, attentions2, attentions3, text):
    # Set the plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    axes[0].hist(attentions1, bins=21, edgecolor='black')
    axes[1].hist(attentions2, bins=21, edgecolor='black')
    axes[2].hist(attentions3, bins=21, edgecolor='black')

    # Add labels and title for each subplot
    axes[0].set_title("Mean aggregation")
    axes[1].set_title("Lenient (OR) aggregation")
    axes[2].set_title("Conservative (AND) aggregation")

    axes[0].set_xlabel("Proportion of zeros")
    axes[0].set_ylabel("Frequency")
    axes[1].set_xlabel("Proportion of zeros")
    axes[1].set_ylabel("Frequency")
    axes[2].set_xlabel("Proportion of zeros")
    axes[2].set_ylabel("Frequency")

    plt.suptitle("Number of zeros value over sequence length", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'Rationales/Zeros/{text}.png')
    plt.close()

