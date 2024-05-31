from utils import *


def aggregate_attention(at_mask, row, params):
    """
    Aggregate attention from different annotators.
    :param at_mask: attention masks of the three annotators
    :param row: dataframe row
    :param params: parameters dictionary
    :return aggregated attention vector
    """
    type_attention = params['type_attention']
    aggregation_attention = params['aggr_attention']
    variance = params['variance']
    at_mask_fin = at_mask

    # If the final label is normal or non-toxic then each value is represented by 1/len(sentences)
    if row['final_label'] in ['normal', 'non-toxic']:
        at_mask_fin = [1 / len(at_mask[0]) for _ in at_mask[0]]
        return at_mask_fin
    elif len(at_mask) > 1:  # Only if there are multiple annotators (this happens always with the HateXplain dataset)
        at_mask_fin = [[value * int(variance) for value in inner_list] for inner_list in at_mask_fin]

        if aggregation_attention == 'mean':
            at_mask_fin = np.mean(np.array(at_mask_fin), axis=0)
        elif aggregation_attention == 'lenient':  # OR attention
            at_mask_fin = 1 * (np.sum(np.array(at_mask_fin), axis=0) > 0)
        elif aggregation_attention == 'conservative':  # AND attention
            at_mask_fin = np.prod(np.array(at_mask_fin), axis=0)
    else:
        at_mask_fin = np.array(at_mask_fin[0])

    if not params['evaluate']:
        # Final vector is normalized (for the sexism dataset this is the only processing that happens)
        if type_attention == 'sigmoid':
            at_mask_fin = sigmoid(at_mask_fin)
        elif type_attention == 'softmax':
            at_mask_fin = softmax(at_mask_fin)
        elif type_attention == 'neg_softmax':
            at_mask_fin = neg_softmax(at_mask_fin)
        elif type_attention in ['raw', 'individual']:
            pass

    if params['decay']:
        at_mask_fin = decay(at_mask_fin, params)

    return at_mask_fin


def distribute(old_distribution, new_distribution, index, left, right, params):
    """
    Distribution function.
    :param old_distribution: previous distribution of attention weights
    :param new_distribution: current new distribution of attention weights
    :param index: position of the tokens considered
    :param left: number of position before to consider
    :param right: number of position after to consider
    :param params: parameters dictionary
    :return: current new distribution of attention weights
    """
    alpha = params['alpha']
    p_value = params['p_value']
    method = params['method']

    # Values of the attention from the old distribution of the given word
    reserve = alpha * old_distribution[index]

    # If the method is additive
    if method == 'additive':  # The value reserve is divided equally among the values to the right and to the left
        for temp in range(index - left, index):  # For the left side
            new_distribution[temp] = new_distribution[temp] + reserve / (left + right)

        for temp in range(index + 1, index + right):  # For the right side
            new_distribution[temp] = new_distribution[temp] + reserve / (left + right)

    elif method == 'geometric':
        # First generate the geometric distribution for the left side
        temp_sum = 0.0
        new_prob = []
        for temp in range(left):  # For the left side
            each_prob = p_value * ((1.0 - p_value) ** temp)  # Geometric distribution
            new_prob.append(each_prob)  # Save the probability of geometric distribution
            temp_sum += each_prob  # Sum the probability
            new_prob = [each / temp_sum for each in new_prob]  # Normalize the probability so they sum up to 1

        for temp in range(index - left, index):
            # The new distribution will have the reserve value multiply by the probability
            # The indexing of the new prob is such that the nearer words get the higher probability
            new_distribution[temp] = new_distribution[temp] + reserve * new_prob[-(temp - (index - left)) - 1]

        # Do the same for right side, but now the order is opposite
        temp_sum = 0.0
        new_prob = []
        for temp in range(right):
            each_prob = p_value * ((1.0 - p_value) ** temp)  # Geometric distribution
            new_prob.append(each_prob)  # Save the probability of geometric distribution
            temp_sum += each_prob  # Sum the probability
            new_prob = [each / temp_sum for each in new_prob]  # Normalize the probability so they sum up to 1

        for temp in range(index + 1, index + right):
            # The new distribution will have the reserve value multiply by the probability
            # The indexing of the new prob is such that the nearer words get the higher probability
            new_distribution[temp] = new_distribution[temp] + reserve * new_prob[temp - (index + 1)]

    return new_distribution


def decay(old_distribution, params):
    """
    Decay the attentions left and right of the attention words.
    This is done to decentralise the attention to a single word.
    :param old_distribution: previous attention weights value
    :param params: parameters' dictionary
    :return: final new values for attention weights
    """
    window = params['window']
    # Define a new distribution with just 0s
    new_distribution = [0.0] * len(old_distribution)
    # For every token in the sentence
    for index in range(len(old_distribution)):
        # Find the number of words after
        right = min(window, len(old_distribution) - index)
        # Find the number of words before
        left = min(window, index)
        # Compute the new distribution
        new_distribution = distribute(old_distribution, new_distribution, index, left, right, params)

    if params['normalized']:
        norm_distribution = []
        for index in range(len(old_distribution)):
            norm_distribution.append(old_distribution[index] + new_distribution[index])
        temp_sum = sum(norm_distribution)
        new_distribution = [each / temp_sum for each in norm_distribution]
    return new_distribution
