import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, AutoTokenizer
from attention import aggregate_attention
from data_preprocessing import *
import numpy as np
from datasets import Dataset
from utils import *
from tqdm import tqdm


def return_mask(row, tokenizer, params):
    """
    Return the tokenization of the words in the sentence and the attention masks of the three annotators.
    :param row: text
    :param tokenizer: tokenizer
    :param params: params
    :return: tokenization of the words in the sentence and the attention masks of the three annotators
    """
    # Extract the text from the post
    max_length = params['max_length']
    text_tokens = row['text']

    # A very rare corner case
    if len(text_tokens) == 0:
        text_tokens = ['dummy']

    # Get the rationales from the annotators
    mask_all = row['rationales']
    mask_all_temp = mask_all

    # If all predicted non-offensive, then there is no rationales, so we add one
    if params['number_rationales'] == 1:
        if len(mask_all_temp) == 0:
            mask_all_temp.append([0] * len(text_tokens))
    else:  # This is the default case from the HateXplain code
        while len(mask_all_temp) != 3:
            mask_all_temp.append([0] * len(text_tokens))

    word_mask_all = []
    word_tokens_all = []

    # For every annotator's rational
    for mask in mask_all_temp:
        if mask[0] == -1:
            mask = [0] * len(mask)

        list_pos = []
        mask_pos = []

        flag = 0
        for i in range(0, len(mask)):  # For every word
            # Find when the mask changes
            if i == 0 and mask[i] == 0:
                list_pos.append(0)
                mask_pos.append(0)
            elif flag == 0 and mask[i] == 1:
                mask_pos.append(1)
                list_pos.append(i)
                flag = 1
            elif flag == 1 and mask[i] == 0:
                flag = 0
                mask_pos.append(0)
                list_pos.append(i)

        if list_pos[-1] != len(mask):
            list_pos.append(len(mask))
            mask_pos.append(0)

        # For every block with the same mask, get text of the block
        string_parts = []
        for i in range(len(list_pos) - 1):
            string_parts.append(text_tokens[list_pos[i]:list_pos[i + 1]])

        # Add [CLS] (which is 101) and give an attention of zero
        word_tokens = [101]
        word_mask = [0]

        # For every block of text with the same mask
        for i in range(0, len(string_parts)):
            # Process the tokens
            tokens = ek_extra_preprocess(" ".join(string_parts[i]), tokenizer)
            # Set the mask to the value of that block
            masks = [mask_pos[i]] * len(tokens)
            # Add the list of tokens
            word_tokens += tokens
            # Add the mask of the corresponding tokens
            word_mask += masks

        # Truncation if necessary
        word_tokens = word_tokens[0:(int(max_length) - 2)]
        word_mask = word_mask[0:(int(max_length) - 2)]
        # Add [SEP] toke (which correspond to 102) and set its mask to 0
        word_tokens.append(102)
        word_mask.append(0)

        word_mask_all.append(word_mask)
        if len(word_tokens_all) == 0:
            word_tokens_all.append(word_tokens)

    if len(mask_all) == 0:
        word_mask_all = []
    else:
        word_mask_all = word_mask_all[0:len(mask_all)]
    # We have three word_tokens lists, so we return only one (they are all the same) and
    # then we return the attention masks
    return word_tokens_all[0], word_mask_all


def get_processed_hatexplain_data(data, tokenizer, params):
    """
    Build a DataFrame with post id, text, attention and label.
    :param data: dataframe with post ids, text, attentions and labels column only
    :param tokenizer: tokenizer used for the tokenization
    :param params: parameter dictionary
    :return: training data in the columns post_id, text, attention and labels
    """
    post_ids_list = []
    text_list = []
    attention_list = []
    label_list = []
    count_confused = 0

    for index, row in tqdm(data.iterrows(), total=len(data)):
        # Extract the final label (computed with the majority)
        annotation = row['final_label']
        # If the annotation is not undecided (so a label exists)
        if annotation != 'undecided':
            # Add the post's id
            post_ids_list.append(row['post_id'])
            # Add the final label
            label_list.append(annotation)
            # Get the tokenization of the words in the sentence and the attention masks of the three annotators
            tokens_all, attention_masks = return_mask(row, tokenizer, params)
            # Get the final attention
            attention_vector = aggregate_attention(attention_masks, row, params)
            # Build the list of attention vectors and the list of word tokens
            attention_list.append(attention_vector)
            text_list.append(tokens_all)
        else:
            count_confused += 1

    print("no_majority:", count_confused)
    # Calling DataFrame constructor after zipping both lists, with columns specified
    training_data = pd.DataFrame(list(zip(post_ids_list, text_list, attention_list, label_list)),
                                 columns=['Post_id', 'Text', 'Attention', 'Label'])

    return training_data


def get_annotated_data(data, params):
    """
    Get information on the posts and choose the correct final label.
    :param data: dataset (could be trained, val or test set)
    :param params: parameters
    :return: dataframe containing information on the posts: post's id, text, rationales, final label, targets
    """
    majority = params['majority']
    num_classes = params['num_classes']

    dict_data = []
    for post in data:  # For each post
        # Get information on the post
        temp = {'post_id': post['id'], 'text': post['post_tokens'], 'rationales': post['rationales']}

        # Compute the final label based on majority
        final_label = []
        for i in range(1, 4):  # For each annotator
            # Get the post information on the annotators id
            temp['annotatorid' + str(i)] = post['annotators']['annotator_id'][i - 1]
            # Get the post information on target groups
            temp['target' + str(i)] = post['annotators']['target'][i - 1]
            # Get the post information on the label
            temp['label' + str(i)] = post['annotators']['label'][i - 1]
            final_label.append(temp['label' + str(i)])

        # Find the most voted label
        final_label_id = max(final_label, key=final_label.count)

        # If the majority voted for a label, set the label. Otherwise, the post is undecided
        if final_label.count(final_label_id) < majority:
            temp['final_label'] = 'undecided'  # undecided
        else:
            if final_label_id == 0:
                temp['final_label'] = 'hatespeech'
            elif final_label_id == 1:
                temp['final_label'] = 'normal'
            elif final_label_id == 2:
                temp['final_label'] = 'offensive'

        # Modify if the number of classes is two
        if num_classes == 2:
            if final_label.count(final_label_id) >= majority:
                if temp['final_label'] in ['hatespeech', 'offensive']:
                    final_label_id = 'toxic'
                else:
                    final_label_id = 'non-toxic'
                temp['final_label'] = final_label_id

        # Add the information to a list
        dict_data.append(temp)

    # Create a DataFrame with the information on the different posts
    temp_read = pd.DataFrame(dict_data)
    return temp_read


def collect_data(dataset, tokenizer, params):
    """
    Returns a DataFrame with post id, text, attention and label.
    :param dataset: dataset (could be trained, val or test set)
    :param tokenizer: tokenizer
    :param params: parameter dictionary
    :return: DataFrame with post id, text, attention and label
    """
    # Get information on the posts and choose the correct final label, return a DataFrame
    data_all_labelled = get_annotated_data(dataset, params)
    # DataFrame with post id, text, attention and label, attention and mask are processed
    train_data = get_processed_hatexplain_data(data_all_labelled, tokenizer, params)
    return train_data


def get_annotated_and_processed_sexism_data(data, tokenizer, params):
    """
    Build a DataFrame with post id, text, attention and label.
    :param data: dataset (could be trained, val or test set)
    :param tokenizer: tokenizer
    :param params: parameters
    :return: dataframe containing information on the posts: post's id, text, rationales, final label, targets
    """
    post_ids_list = []
    text_list = []
    attention_list = []
    label_list = []

    for post in data:  # For each post
        # Get information on the post
        temp = {'post_id': post['id'], 'text': post['post_tokens'], 'rationales': post['rationales'],
                'final_label': post['label']}

        # Add the post's id
        post_ids_list.append(temp['post_id'])
        # Add the final label
        label_list.append(temp['final_label'])
        # Get the tokenization of the words in the sentence and the attention masks of the three annotators
        tokens_all, attention_masks = return_mask(temp, tokenizer, params)
        # Get the final attention
        attention_vector = aggregate_attention(attention_masks, temp, params)
        # Build the list of attention vectors and the list of word tokens
        attention_list.append(attention_vector)
        text_list.append(tokens_all)

    # Calling DataFrame constructor after zipping both lists, with columns specified
    training_data = pd.DataFrame(list(zip(post_ids_list, text_list, attention_list, label_list)),
                                 columns=['Post_id', 'Text', 'Attention', 'Label'])

    return training_data


def encode_data(dataframe):
    """
    For each sample, return the text, the attention (from the annotators) and the labels.
    :param dataframe: dataframe
    :return: a list with, for each sample, text, the attention (from the annotators) and the labels
    """
    tuple_new_data = []
    for index, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        tuple_new_data.append((row['Text'], row['Attention'], row['Label']))
    return tuple_new_data


def set_tokenizer(params):
    """
    Set the tokenizer.
    :param params: parameters' dictionary
    :return: tokenizer
    """
    # Set tokenizer based on the model
    tokenizer = None
    if params['tokenizer'] == 'default':
        if params['model'] in ['bert', 'hatexplain', 'bert_mlp']:
            print('Loading BERT tokenizer...')
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False)
        elif params['model'] == 'hatebert':
            print('Loading HateBERT tokenizer...')
            tokenizer = AutoTokenizer.from_pretrained("GroNLP/hateBERT")

    elif params['tokenizer'] == 'bert_miniature':
        print('Loading BERT miniature tokenizer...')
        tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-12_H-768_A-12",
                                                  model_max_length=params['max_length'])
    return tokenizer


def process_data_hatexplain(train, val, test, tokenizer, params):
    """
    Return the final list for the training, validation and test set.
    :param train: dataset with all the train samples
    :param val: dataset with all the validation samples
    :param test: dataset with all the test samples
    :param tokenizer: tokenizer
    :param params: parameter dictionary
    :return: train, val and test sets
    """
    # DataFrame with post id, text, attention and label
    X_train = collect_data(train, tokenizer, params)
    X_val = collect_data(val, tokenizer, params)
    X_test = collect_data(test, tokenizer, params)

    # A list with, for each sample, text, the attention (from the annotators) and the labels
    X_train = encode_data(X_train)
    X_val = encode_data(X_val)
    X_test = encode_data(X_test)

    print("total dataset size:", len(X_train) + len(X_val) + len(X_test))
    return X_train, X_val, X_test


def process_data_sexism(train, tokenizer, params):
    """
    Return the final list for the training set.
    :param train: dataset with all the train samples
    :param tokenizer: tokenizer
    :param params: parameter dictionary
    :return: train, val and test sets
    """
    # DataFrame with post id, text, attention and label
    X_train = get_annotated_and_processed_sexism_data(train, tokenizer, params)

    # A list with, for each sample, text, the attention (from the annotators) and the labels
    X_train = encode_data(X_train)

    print("total dataset size:", len(X_train))
    return X_train


def get_original_tweets(data, datasets=['benevolent', 'hostile', 'other'], balance_classes=True):
    """
    Get tweets in the sexism dataset that are not adversarial examples.
    :param data: dataset containing original and adversarial tweets, is an aggregation of different sub datasets
    :param datasets: list of string: names of the subset of datasets from which the tweets are selected.
    default = ['benevolent', 'hostile', 'other']
    :param balance_classes: bool, whether to have the same number of sexist and non-sexist tweets. default=True
    :return: dataset of original tweets
    """
    # drop adversarial examples
    originals = data[data['of_id'] == -1]
    originals = originals[originals.dataset.isin(datasets)]

    # equalize class sizes
    if balance_classes:
        min_class_size = originals.groupby("sexist").size().min()
        originals = originals.groupby("sexist").apply(lambda x: x.sample(n=min_class_size,
                                                                         replace=False)).reset_index(0, drop=True)
    originals = originals.drop(['of_id'], axis=1)

    return originals


def get_adversarial_examples(originals, data):
    """
    Get one adversarial example corresponding to each tweet in the original dataset.
    :param originals: dataset of tweets that are not adversarial examples.
    :param data: dataset containing original and adversarial tweets, is an aggregation of different sub-datasets
    :return: dataset of adversarial tweets
    """
    # only pick non-sexist modifications
    modifications = data[(data['sexist'] == False) & (data['of_id'] != -1)]

    # pick at most 1 modification for each tweet in originals
    modifications = modifications[modifications.of_id.isin(originals['id'].values)]
    modifications = modifications.groupby("of_id").apply(lambda x: x.sample(n=1)).reset_index(0, drop=True)

    return modifications


def drop_originals_without_adversarial(originals, modifications):
    """
    Removes the original tweets that do not have a corresponding adversarial example to equalize adversarial examples
    and original tweets. If needed, removes tweets to equalize sexists and non-sexist tweets.
    :param originals: dataset of tweets that are not adversarial examples
    :param modifications: dataset of adversarial examples
    """
    originals.sexist = originals.sexist.astype(np.bool_)
    modifications.sexist = modifications.sexist.astype(np.bool_)
    modifications['of_id'] = modifications['of_id'].astype(int)
    originals['id'] = originals['id'].astype(int)

    # drop originals that don't have a modification -> balances original/modifications
    missing_modification_filter = (~originals.id.isin(modifications.of_id.unique())) & originals.sexist
    originals.drop(originals[missing_modification_filter].index, inplace=True)

    # remove as many non-sexist examples as sexist examples dropped -> keep the sexist/non-sexist balance
    originals.drop(originals[~originals.sexist].sample(n=missing_modification_filter.sum(), replace=False).index,
                   inplace=True)


def get_one_split(originals, modifications, test_frac=.1):
    """
    Splits the original tweets and the adversarial examples in balanced train and test sets.
    :param originals: dataset of tweets that are not adversarial examples
    :param modifications: dataset of adversarial examples
    :param test_frac: float, fraction of samples to put in the test set
    :return: train and test set of original tweets and train and test set of adversarial examples
    """
    # set id as index if it exists
    try:
        originals.set_index("id", inplace=True)
    except:
        pass

    # sample a fraction of sexist and non-sexist examples
    originals_test = \
        originals.groupby("sexist").apply(lambda x: x.sample(frac=test_frac, replace=False)).reset_index(0, drop=True)
    # select samples not in test set to create train set
    originals_train = originals[~originals.index.isin(originals_test.index)].sample(frac=1.)

    # select the adversarial test examples that correspond to the original test set samples
    tomodify_test = originals_test[originals_test.sexist].index
    modifications_test = pd.concat((originals_test[originals_test.sexist],
                                    modifications[modifications.of_id.isin(tomodify_test)]), sort=False)

    # sample 50% of indices of the original sexist tweets
    tomodify_train = originals_train[originals_train.sexist].sample(frac=.5).index
    # select adversarial examples corresponding to above ids and drop same number of non-sexist rows to create the
    # adversarial train set
    modifications_train = pd.concat(
        (originals_train.drop(originals_train[~originals_train.sexist].sample(n=len(tomodify_train)).index),
         modifications[modifications.of_id.isin(tomodify_train)]), sort=False).sample(frac=1.)

    return originals_train, originals_test, modifications_train, modifications_test


def stratified_sample_df(df, col, n_samples):
    """
    Create a stratified sample of a dataset.
   :param df: dataset to sample from
   :param col: column to stratify by
   :param n_samples: desired number of samples per group
   :return: Stratified sample dataset
   """
    # determine sample size per group (limited by smallest group size)
    n = min(n_samples, df[col].value_counts().min())
    # sample n items from each group
    df_ = df.groupby(col).apply(lambda x: x.sample(n))
    # drop the added group level from the index
    df_.index = df_.index.droplevel(0)

    return df_


def split_data_sexism(data):
    """
    Split data into train and test sets. The sets are balanced with regards: to sexist/non-sexist samples,
    adversarial/original tweets and type of initial subdataset the tweets come from.
    :param data: dataset containing original and adversarial tweets, is an aggregation of different subdatasets
    :return: train and test set
    """

    # get original tweets and adversarial examples from benevolent, hostile and other subdataset
    originals_reproduction = get_original_tweets(datasets=['benevolent', 'hostile', 'other'], data=data)
    modifications_reproduction = get_adversarial_examples(originals_reproduction, data=data)
    # drop original tweets that do not correspond to an adversarial example
    drop_originals_without_adversarial(originals_reproduction, modifications_reproduction)

    # get original tweets and adversarial examples from callme subdataset
    originals_replication = get_original_tweets(datasets=['callme'], data=data)
    modifications_replication = get_adversarial_examples(originals_replication, data=data)
    # drop original tweets that do not correspond to an adversarial example
    drop_originals_without_adversarial(originals_replication, modifications_replication)

    # get original tweets and adversarial examples from benevolent and hostile subdataset
    originals_bh = get_original_tweets(datasets=['benevolent', 'hostile'], data=data)
    modifications_bh = get_adversarial_examples(originals_bh, data=data)
    # drop original tweets that do not correspond to an adversarial example
    drop_originals_without_adversarial(originals_bh, modifications_bh)

    # get original tweets from scales subdataset
    originals_goldtrain = get_original_tweets(datasets=['scales'], data=data)
    # create empty df since scales doesn't have any adversarial examples
    modifications_goldtrain = pd.DataFrame(columns=modifications_bh.columns)

    # split goldtrain dataset in train and test sets
    originals_train_goldtrain, originals_test_goldtrain, _, _ = get_one_split(originals_goldtrain, modifications_goldtrain)

    # split replication dataset in original and adversarial train and test sets
    originals_train_replication, originals_test_replication, modifications_train_replication, \
        modifications_test_replication = get_one_split(originals_replication, modifications_replication)

    # split reproduction dataset in original and adversarial train and test sets
    originals_train_reproduction, originals_test_reproduction, modifications_train_reproduction, \
        modifications_test_reproduction = get_one_split(originals_reproduction, modifications_reproduction)

    # concatenate all train sets in a balanced train set
    train = pd.concat((stratified_sample_df(modifications_train_reproduction, "sexist", 300,),
                       stratified_sample_df(modifications_train_replication, "sexist", 100),
                       stratified_sample_df(originals_train_goldtrain, "sexist", 100),
                       ), sort=False).sample(frac=1.)

    # concatenate all test sets in a balanced test set
    test = pd.concat((stratified_sample_df(modifications_test_reproduction, "sexist", 300),
                      stratified_sample_df(modifications_test_replication, "sexist", 100),
                      stratified_sample_df(originals_test_goldtrain, "sexist", 100),
                      ), sort=False)

    return train, test


def df_to_torch_dataset(dataframe):
    """
    Transform dataframe to torch dataset.
    :param dataframe: dataframe to convert
    :return: torch dataset
    """
    dataframe.reset_index(drop=True, inplace=True)
    dataset = Dataset.from_pandas(dataframe)
    dataset.set_format('torch')

    return dataset


def process_data_distillation(data, tokenizer, params):
    """
    Processes and return the final test and training set.
    :param data: dataset containing original and adversarial tweets, is an aggregation of different subdatasets
    :param tokenizer: tokenizer used for the tokenization
    :param params: parameter dictionary
    :return:
    """

    # add columns to the dataset to store the tokenization data
    data['input_ids'] = None
    data['token_type_ids'] = None
    data['attention_mask'] = None
    # tokenize text and add the information to the dataset
    for idx, row in data.iterrows():
        tokenized_data = tokenizer(row['text'], padding='max_length', max_length=params['max_length'], truncation=True)
        data.at[idx, 'input_ids'] = tokenized_data['input_ids']
        data.at[idx, 'token_type_ids'] = tokenized_data['token_type_ids']
        data.at[idx, 'attention_mask'] = tokenized_data['attention_mask']

    # split data
    data_train, data_test = split_data_sexism(data)

    # format datasets for pytorch
    train = df_to_torch_dataset(data_train[['labels', 'input_ids', 'token_type_ids', 'attention_mask']])
    test = df_to_torch_dataset(data_test[['labels', 'input_ids', 'token_type_ids', 'attention_mask']])

    return train, test
