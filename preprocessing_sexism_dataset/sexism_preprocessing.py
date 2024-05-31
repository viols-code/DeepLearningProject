import pandas as pd
import difflib
import re
import json

def preprocess_text(text, url_pattern =r'https?://\S+|www\.\S+',
                    user_pattern = r'\b\w*MENTION\w*\b',
                    percent_pattern = r'\b\d+(?:\.\d+)?%\b',
                    date_pattern = r'\b\d{4}\b',
                    number_pattern = r'\b\d+(?:\.\d+)?(?:\s*\w+)\b',
                    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                    happy_pattern = r':\)',
                    sad_pattern = r':\(',
                    wink_pattern = r';\)'):
    """
    Add the tags: <user>, <url>, <percent>, <date>, <number>, <email>, <happy>, <sad>, <wink>
    to replace the corresponding pattern in arguments.
    Separate the text into tokens and adds them to a new column of the dataframe.
    :param text: string, text to pre-process
    :return: string, text after preprocessing
    """
    text = text.apply(lambda t: re.sub(user_pattern, '<user>', t))
    text = text.apply(lambda t: re.sub(url_pattern, '<url>', t))
    text = text.apply(lambda t: re.sub(percent_pattern, '<percent>', t))
    text = text.apply(lambda t: re.sub(date_pattern, '<date>', t)) # only yyyy format
    text = text.apply(lambda t: re.sub(number_pattern, '<number>', t))
    text = text.apply(lambda t: re.sub(email_pattern, '<email>', t))
    text = text.apply(lambda t: re.sub(happy_pattern, '<happy>', t))
    text = text.apply(lambda t: re.sub(sad_pattern, '<sad>', t))
    text = text.apply(lambda t: re.sub(wink_pattern, '<wink>', t))

    return text


def tokenize_data(data, sep=' '):
    """
    Separate the words in data['text'] and add them in a new column of the dataframe.
    :param data: pandas dataframe
    :param sep: string, separator to use to tokenize the text
    :return: pandas dataframe
    """
    tokens = []
    for index in data.index:
        tokenized_texts = (data['text'])[index].split(sep)
        tokens.append(tokenized_texts)
    data['tokens'] = tokens

    return data


def compare_token_lists(tokens1, tokens2):
    """
    Compares two lists of tokens and returns a list of 0 and 1s of the same size as tokens1.
    Each element is 0 if the token is the same in both list and a 1 if the token of tokens1
    does not correspond to the token of tokens2 at this position.
    :param tokens1: list of string
    :param tokens2: list of string
    :return: list of length len(tokens1) containing 0s and 1s
    """
    matcher = difflib.SequenceMatcher(None, tokens1, tokens2)
    differences = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # Both lists have the same tokens
            differences.extend([0] * (i2 - i1))
        elif tag == 'replace':
            # Tokens are different between the two lists
            differences.extend([1] * (i2 - i1))
        elif tag == 'delete':
            # Tokens present in tokens1 but not in tokens2
            differences.extend([1] * (i2 - i1))
        elif tag == 'insert':
            # Tokens present in tokens2 but not in tokens1
            differences.extend([1] * (j2 - j1))

    return differences


def get_rationales(data):
    """
    Generate rationales for each tweet in the dataset.
    :param data: Dataframe containing tweet data
    :return: a list of rationales for each tweet in the dataset
    """
    rationales = []
    for idx, tweet in data.iterrows():
        if tweet['of_id'] != -1:
            rationales.append([])
        else:
            adv_tweets = data[data['of_id'] == idx]
            temp = []
            for _, adv_tweet in adv_tweets.iterrows():
                temp.append(compare_token_lists(tweet['tokens'], adv_tweet['tokens']))
            rationales.append(temp)

    return rationales


if __name__ == '__main__':
    data_path = 'Data/'
    raw_data = pd.read_csv(data_path + 'sexism_data.csv')

    # Select data with adversarial examples
    adv_data = raw_data[raw_data['of_id'] != -1]
    ids_to_keep = list(adv_data['of_id']) + list(adv_data['id'])
    adv_data = raw_data[raw_data['id'].isin(ids_to_keep)]

    # Data cleaning
    adv_data = adv_data.set_index('id')
    adv_data = adv_data.rename(columns={"sexist": "label"})
    adv_data['label'] = adv_data['label'].astype(int)

    adv_data['text'] = preprocess_text(adv_data['text'])
    preprocessed = tokenize_data(adv_data)
    preprocessed['rationales'] = get_rationales(preprocessed)

    # Store in a similar format as HateXplain dataset
    result_dict = {}
    for post_id, group in preprocessed.groupby(level=0):
        result_dict[post_id] = {
            'id': str(post_id),
            'label': 'toxic' if group['label'].values[0].tolist() == 1 else 'non-toxic',
            'target': ['Women'],
            'rationales': group['rationales'].values[0],
            'post_tokens': group['tokens'].values[0]
        }

    # Save data as JSON to be able to use load_dataset later
    with open(data_path + "sexism_data_preprocessed.json", "w") as file:
        for key, value in result_dict.items():
            json.dump(value, file)
            file.write("\n")
