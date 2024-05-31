import string
import nltk
from nltk.corpus import stopwords

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
import re


text_processor_bert = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
               'time', 'date', 'number'],
    # terms that will be annotated
    fix_html=True,  # fix HTML tokens
    annotate={"hashtag", "allcaps", "elongated", "repeated",
              'emphasis', 'censored'},
    # corpus from which the word statistics are going to be used
    # for word segmentation
    segmenter="twitter",

    # corpus from which the word statistics are going to be used
    # for spell correction
    # corrector="twitter",

    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words

    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,

    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionary.
    dicts=[emoticons]
)


def custom_tokenize(sent, tokenizer, max_length=512):
    """
    Tokenize the given sentence.
    :param sent: words in the sentence to tokenize
    :param tokenizer: tokenizer
    :param max_length: max length of the result of the tokenization
    :return: tokens mapped to their ids
    """
    try:
        encoded_sent = tokenizer.encode(
            sent,  # Sentence to encode.
            add_special_tokens=False,  # Add '[CLS]' and '[SEP]'
            # max_length = max_length,
            # This function also supports truncation and conversion
            # to pytorch tensors, but we need to do padding, so we
            # can't use these features
            # max_length = 128,          # Truncate all sentences.
            # return_tensors = 'pt',     # Return pytorch tensors.
        )
    except ValueError:
        encoded_sent = tokenizer.encode(
            ' ',  # Sentence to encode.
            add_special_tokens=False,  # Add '[CLS]' and '[SEP]'
            max_length=max_length,
        )
    return encoded_sent


def ek_extra_preprocess(text, tokenizer):
    """
    Process the given text with ekphrasis processor.
    :param text: text to be processed
    :param tokenizer: tokenizer to be used
    :return: list of tokens
    """
    # List of words to remove
    remove_words = ['<allcaps>', '</allcaps>', '<hashtag>', '</hashtag>', '<elongated>', '<emphasis>', '<repeated>',
                    '\'', 's']
    # Preprocess the text with ekphrasis processor
    word_list = text_processor_bert.pre_process_doc(text)
    # Remove the words in the list of words to remove
    word_list = list(filter(lambda a: a not in remove_words, word_list))

    # Join with the space the remaining words
    sent = " ".join(word_list)
    # Replace < or > with a space
    sent = re.sub(r"[<*>]", " ", sent)
    sub_word_list = custom_tokenize(sent, tokenizer)
    # Return the list of tokens
    return sub_word_list


def strip_links(text):
    """
    Removes url from text and replaces them with a comma and a space.
    :param text: text to process
    :return: text without urls
    """
    # define url pattern
    link_regex = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links = re.findall(link_regex, text)
    # replace urls
    for link in links:
        text = text.replace(link[0], ', ')

    return text


def strip_all_entities(text, stopwords):
    """
    Lower cases the text, removes urls, punctuation and stopwords, replaces mentions and hashtags by 'UNK'.
    :param text: text to process
    :param stopwords: list of stopwords
    :return: modified text
    """
    # remove urls
    text = strip_links(text)

    # replaces all separators by a space
    entity_prefixes = ['@', '#']
    for separator in string.punctuation:
        if separator not in entity_prefixes:
            text = text.replace(separator, ' ')

    words = []
    # for each word
    for word in text.split():
        word = word.strip()
        if word:
            # keep the word if it's not a hashtag, a mention
            if word[0] not in entity_prefixes and word not in stopwords:
                words.append(word.lower())
            # if the word is a stopword, don't keep it
            elif word in stopwords:
                continue
            # if the word is a mention or a hashtag, replace it by 'UNK'
            else:
                words.append('UNK')
    return ' '.join(words)


def indira_preprocess(data):
    """
    Preprocessing data.
    :param data: dataset containing original and adversarial tweets, is an aggregation of different sub-datasets
    :return: preprocessed data with labels
    """
    # fill unknown labels with False
    data['sexist'] = data['sexist'].fillna(False)
    # remove data with unknown toxicity
    data = data[data['toxicity'].notna()]

    # pre-process the text
    nltk.download('stopwords')
    stops = set(stopwords.words('english'))
    stops.discard('not')
    for idx, row in data.iterrows():
        data.at[idx, 'text'] = strip_all_entities(row['text'], stops)

    # add labels
    data['labels'] = (data['sexist'] == True).astype(int)

    return data
