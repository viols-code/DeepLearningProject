from sklearn.metrics import roc_auc_score


def calculate_bias_metrics(dataset, method, community, classification):
    """
    Compute the type of bias for the given community on the given dataset.
    :param dataset: dataset to consider for the bias computation
    :param method: can be subgroup, bpsn or bnsp
    :param community: given community to consider for the bias computation
    :param classification: probabilities of classification
    :return:
    """
    positive_ids = []
    negative_ids = []

    if method == 'subgroup':
        for index, row in dataset.iterrows():
            count = sum(community in sublist for sublist in [row['target1'], row['target2'], row['target3']])
            if count >= 2:
                if row['final_label'] in ['normal', 'non-toxic']:
                    negative_ids.append([row['post_id'], row['final_label'], classification[index]])
                else:
                    positive_ids.append([row['post_id'], row['final_label'], classification[index]])
            else:
                pass
    elif method == 'bpsn':
        for index, row in dataset.iterrows():
            count = sum(community in sublist for sublist in [row['target1'], row['target2'], row['target3']])
            if count >= 2:
                if row['final_label'] in ['normal', 'non-toxic']:
                    negative_ids.append([row['post_id'], row['final_label'], classification[index]])
                else:
                    pass
            else:
                if row['final_label'] not in ['normal', 'non-toxic']:
                    positive_ids.append([row['post_id'], row['final_label'], classification[index]])
                else:
                    pass
    elif method == 'bnsp':
        for index, row in dataset.iterrows():
            count = sum(community in sublist for sublist in [row['target1'], row['target2'], row['target3']])
            if count >= 2:
                if row['final_label'] not in ['normal', 'non-toxic']:
                    positive_ids.append([row['post_id'], row['final_label'], classification[index]])
                else:
                    pass
            else:
                if row['final_label'] in ['normal', 'non-toxic']:
                    negative_ids.append([row['post_id'], row['final_label'], classification[index]])
                else:
                    pass
    else:
        print('Incorrect option selected!!!')

    return {'positiveID': positive_ids, 'negativeID': negative_ids}


def get_bias_dict(method_list, community_list, bias_test_data, classification):
    """
    Computes ROC AUC score for each community for each method and each community.
    :param method_list: list of methods
    :param community_list: list of communities
    :param bias_test_data: test data to consider for the bias computation
    :param classification: probabilities of classification
    :return:
    """
    for each_method in method_list:
        for each_community in community_list:
            community_data = calculate_bias_metrics(bias_test_data, each_method, each_community, classification)
            truth_values = []
            prediction_values = []

            for each in community_data['positiveID']:
                truth_values.append(1)
                prediction_values.append(each[2]['toxic'])

            for each in community_data['negativeID']:
                truth_values.append(0)
                prediction_values.append(each[2]['toxic'])

            roc_output_value = roc_auc_score(truth_values, prediction_values)
            print(each_method)
            print(each_community)
            print(roc_output_value)