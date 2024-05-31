import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, precision_score
from sklearn.utils import class_weight
import json
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import pandas as pd
import numpy as np
from utils import *
from data_processing import *
from train import *


def convert_data(test_data, list_dict, rational_present=True, topk=2):
    """this converts the data to be with or without the rationals based on the previous predictions"""
    """input: params -- input dict, list_dict -- previous predictions containing rationals
    rational_present -- whether to keep rational only or remove them only
    topk -- how many words to select"""

    temp_dict = {}
    for ele in list_dict:
        temp_dict[ele['annotation_id']] = ele['rationales'][0]['soft_rationale_predictions']

    test_data_modified = []

    for index, row in tqdm(test_data.iterrows(), total=len(test_data)):
        try:
            attention = temp_dict[row['Post_id']]
        except KeyError:
            continue
        topk_indices = sorted(range(len(attention)), key=lambda i: attention[i])[-topk:]
        new_text = []
        new_attention = []
        if rational_present:
            new_attention = [0]
            new_text = [101]

            for i in range(len(row['Text'])):
                if i in topk_indices:
                    new_text.append(row['Text'][i])
                    new_attention.append(row['Attention'][i])
            new_attention.append(0)
            new_text.append(102)

        else:
            for i in range(len(row['Text'])):
                if i not in topk_indices:
                    new_text.append(row['Text'][i])
                    new_attention.append(row['Attention'][i])
        test_data_modified.append([row['Post_id'], new_text, new_attention, row['Label']])

    df = pd.DataFrame(test_data_modified, columns=test_data.columns)
    return df


def standalone_eval_with_rational(params, test, model, tokenizer, topk=2, use_ext_df = False, test_data = None):
    if use_ext_df:
        test = collect_data(test, tokenizer, params)
        test_extra = encode_data(test_data)
        test_dataloader = combine_features(test_extra, params, is_train=False)
    else:
        test = collect_data(test, tokenizer, params)
        test_extra = encode_data(test)
        test_dataloader = combine_features(test_extra, params, is_train=False)

    device = torch.device("cpu")

    if params['auto_weights']:
        y_test = [ele[2] for ele in test]
        params['weights'] = class_weight.compute_class_weight(class_weight='balanced',
                                                              classes=np.unique(y_test), y=y_test).astype('float32')
    else:
        params['weights'] = [1.0] * params['num_classes']

    model.eval()
    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    # Tracking variables
    post_id_all = list(test['Post_id'])

    print("Running eval on test data...")
    true_labels = []
    pred_labels = []
    logits_all = []
    attention_all = []
    input_mask_all = []

    # Evaluate data for one epoch
    for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):

        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention vals
        #   [2]: attention mask
        #   [3]: labels
        b_input_ids = batch[0].to(device)
        b_att_val = batch[1].to(device)
        b_input_mask = batch[2].to(device)
        b_labels = batch[3].to(device)

        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        # model.zero_grad()
        outputs = model(b_input_ids, attention_mask=b_input_mask)
        outputs = compute_loss(outputs, params, attention_mask=b_input_mask, attention_vals=b_att_val,
                               labels=None)

        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.detach().cpu().numpy()

        attention_vectors = np.mean(outputs[1][11][:, :, 0, :].detach().cpu().numpy(), axis=1)

        # Calculate the accuracy for this batch of test sentences.
        # Accumulate the total accuracy.
        pred_labels += list(np.argmax(logits, axis=1).flatten())
        true_labels += list(label_ids.flatten())
        logits_all += list(logits)
        attention_all += list(attention_vectors)
        input_mask_all += list(batch[2].detach().cpu().numpy())

    logits_all_final = []
    for logits in logits_all:
        logits_all_final.append(softmax(logits))

    if use_ext_df == False:
        testf1 = f1_score(true_labels, pred_labels, average='macro')
        testacc = accuracy_score(true_labels, pred_labels)
        # testrocauc=roc_auc_score(true_labels, logits_all_final,multi_class='ovo',average='macro')
        testprecision = precision_score(true_labels, pred_labels, average='macro')
        testrecall = recall_score(true_labels, pred_labels, average='macro')

        # Report the final accuracy for this validation run.
        print(" Accuracy: {0:.3f}".format(testacc))
        print(" Fscore: {0:.3f}".format(testf1))
        print(" Precision: {0:.3f}".format(testprecision))
        print(" Recall: {0:.3f}".format(testrecall))
        # print(" Roc Auc: {0:.3f}".format(testrocauc))

    attention_vector_final = []
    for x, y in zip(attention_all, input_mask_all):
        temp = []
        for x_ele, y_ele in zip(x, y):
            if y_ele == 1:
                temp.append(x_ele)
        attention_vector_final.append(temp)

    list_dict = []

    for post_id, attention, logits, pred, ground_truth in zip(post_id_all, attention_vector_final, logits_all_final,
                                                              pred_labels, true_labels):
        if (ground_truth == 1 and params['num_classes'] == 3) or (ground_truth == 0 and params['num_classes'] == 2):
            continue
        temp = {}

        encoder = LabelEncoder()
        if params['num_classes'] == 2:
            encoder.classes_ = np.array(['non-toxic', 'toxic'])
        elif params['num_classes'] == 3:
            encoder.classes_ = np.array(['hatespeech', 'normal', 'offensive'])

        pred_label = encoder.inverse_transform([pred])[0]
        ground_label = encoder.inverse_transform([ground_truth])[0]

        temp["annotation_id"] = post_id
        temp["classification"] = pred_label
        if params['num_classes'] == 2:
            temp["classification_scores"] = {"non-toxic": logits[0], "toxic": logits[1]}
        elif params['num_classes'] == 3:
            temp["classification_scores"] = {"hatespeech": logits[0], "normal": logits[1], "offensive": logits[2]}

        topk_indicies = sorted(range(len(attention)), key=lambda i: attention[i])[-topk:]

        temp_hard_rationales = []
        for ind in topk_indicies:
            temp_hard_rationales.append({'end_token': ind + 1, 'start_token': ind})

        temp["rationales"] = [{"docid": post_id,
                               "hard_rationale_predictions": temp_hard_rationales,
                               "soft_rationale_predictions": attention,
                               # "soft_sentence_predictions":[1.0],
                               "truth": ground_truth}]
        list_dict.append(temp)

    return list_dict, test

def get_final_dict_with_rational(params, test, model, tokenizer, topk=5):
    list_dict_org, test_data = standalone_eval_with_rational(params, test, model, tokenizer, topk=topk)
    test_data_with_rational = convert_data(test_data, list_dict_org, rational_present=True, topk=topk)
    list_dict_with_rational, _ = standalone_eval_with_rational(params, test, model, tokenizer, topk=topk,
                                                              use_ext_df=True, test_data=test_data_with_rational)
    test_data_without_rational = convert_data(test_data, list_dict_org, rational_present=False, topk=topk)
    list_dict_without_rational, _ = standalone_eval_with_rational(params, test, model, tokenizer,
                                                                  test_data=test_data_without_rational,
                                                                  topk=topk, use_ext_df=True)

    final_list_dict = []
    for ele1, ele2, ele3 in zip(list_dict_org, list_dict_with_rational, list_dict_without_rational):
        ele1['sufficiency_classification_scores'] = ele2['classification_scores']
        ele1['comprehensiveness_classification_scores'] = ele3['classification_scores']
        final_list_dict.append(ele1)
    return final_list_dict

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):  #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_explainability(params, test_data, model, tokenizer):
    set_the_random()
    final_list_dict = get_final_dict_with_rational(params, test_data, model, tokenizer, topk=5)

    path_name = params['model']
    path_name_explanation = 'explanations_dicts/' + path_name + '_' + str(
        params['att_lambda']) + '_explanation_top5.json'
    with open(path_name_explanation, 'w') as fp:
        fp.write('\n'.join(json.dumps(i, cls=NumpyEncoder) for i in final_list_dict))