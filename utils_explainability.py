from train import *
from data_processing import *
import json
import more_itertools as mit


# https://stackoverflow.com/questions/2154249/identify-groups-of-continuous-numbers-in-a-list
def find_ranges(iterable):
    """Yield range of consecutive numbers."""
    for group in mit.consecutive_groups(iterable):
        group = list(group)
        if len(group) == 1:
            yield group[0]
        else:
            yield group[0], group[-1]


# Convert dataset into ERASER format: https://github.com/jayded/eraserbenchmark/blob/master/rationale_benchmark/utils.py
def get_evidence(post_id, anno_text, explanations):
    output = []

    indexes = sorted([i for i, each in enumerate(explanations) if each == 1])
    span_list = list(find_ranges(indexes))

    for each in span_list:
        if type(each) == int:
            start = each
            end = each + 1
        elif len(each) == 2:
            start = each[0]
            end = each[1] + 1
        else:
            print('error')

        output.append({"docid": post_id,
                       "end_sentence": -1,
                       "end_token": end,
                       "start_sentence": -1,
                       "start_token": start,
                       "text": ' '.join([str(x) for x in anno_text[start:end]])})
    return output


# To use the metrices defined in ERASER, we will have to convert the dataset
def convert_to_eraser_format(dataset, method, save_split, save_path, split):
    final_output = []

    if save_split:
        train_fp = open(os.path.join(save_path, 'train.jsonl'), 'w')
        val_fp = open(os.path.join(save_path, 'val.jsonl'), 'w')
        test_fp = open(os.path.join(save_path, 'test.jsonl'), 'w')

    for tcount, eachrow in enumerate(dataset):
        temp = {}
        post_id = eachrow[0]
        post_class = eachrow[1]
        anno_text_list = eachrow[2]
        majority_label = eachrow[1]

        if majority_label == 'normal' or majority_label == 'non-toxic':
            continue

        all_labels = eachrow[4]
        explanations = []
        for each_explain in eachrow[3]:
            explanations.append(list(each_explain))

        # For this work, we have considered the union of explanations. Other options could be explored as well.
        if method == 'union':
            final_explanation = [any(each) for each in zip(*explanations)]
            final_explanation = [int(each) for each in final_explanation]

        temp['annotation_id'] = post_id
        temp['classification'] = post_class
        temp['evidences'] = [get_evidence(post_id, list(anno_text_list), final_explanation)]
        temp['query'] = "What is the class?"
        temp['query_type'] = None
        final_output.append(temp)

        if save_split:
            docs_dir = os.path.join(save_path, 'docs')
            if not os.path.exists(docs_dir):
                os.makedirs(docs_dir)

            with open(os.path.join(docs_dir, post_id), 'w+') as fp:
                fp.write(' '.join([str(x) for x in list(anno_text_list)]))

            if split == 'train':
                train_fp.write(json.dumps(temp) + '\n')
            elif split == 'val':
                val_fp.write(json.dumps(temp) + '\n')
            elif split == 'test':
                test_fp.write(json.dumps(temp) + '\n')
            else:
                print(post_id)

    if save_split:
        test_fp.close()

    return final_output


# Load the whole dataset and get the tokenwise rationales
def get_training_data(data, params, tokenizer):
    final_binny_output = []
    print('total_data', len(data))
    for index, row in tqdm(data.iterrows(), total=len(data)):
        annotation = row['final_label']
        text = row['text']
        post_id = row['post_id']
        annotation_list = [row['label1'], row['label2'], row['label3']]
        tokens_all = list(row['text'])

        if annotation != 'undecided':
            tokens_all, attention_masks = return_mask(row, tokenizer, params)
            final_binny_output.append([post_id, annotation, tokens_all, attention_masks, annotation_list])

    return final_binny_output