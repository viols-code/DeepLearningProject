from datasets import load_dataset
import tensorflow
from keras.preprocessing.sequence import pad_sequences
from transformers import BertForSequenceClassification
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import transformers
from sklearn.utils import class_weight
import random
import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, precision_score
from loss import *
from utils import *
from data_processing import *
from teacher_model import * # file from HateXplain's HugginFace
from Pretrained_Model.bert_mlp import *
from sklearn.preprocessing import LabelEncoder


def save_bert_model(model, tokenizer, params):
    """
    Save model.
    :param model: model
    :param tokenizer: tokenizer
    :param params: parameters
    """
    # Output directory
    output_dir = 'Saved/' + params['path_files'] + '_'
    if params['train_att']:
        if params['att_lambda'] >= 1:
            params['att_lambda'] = int(params['att_lambda'])

        output_dir += str(params['model']) + str(params['supervised_layer_pos']) + '_' + \
                      str(params['num_supervised_heads']) + '_' + str(params['num_classes']) + '_' + \
                      str(params['att_lambda']) + '/'

    else:
        output_dir = output_dir + '_' + str(params['num_classes']) + '/'

    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving model to %s" % output_dir)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def custom_att_masks(input_ids):
    """
    Compute the attention masks.
    :param input_ids: IDs of the input tokens
    :return: attention masks
    """
    attention_masks = []

    for sent in input_ids:  # For each sentence
        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0
        #   - If a token ID is > 0, then it's a real token, set the mask to 1
        att_mask = [int(token_id > 0) for token_id in sent]

        # Store the attention mask for this sentence
        attention_masks.append(att_mask)
    return attention_masks


def return_dataloader(input_ids, labels, att_vals, att_masks, params, is_train=False):
    """
    Return DataLoader with the given batch size.
    :param input_ids: pad sequences of input IDs
    :param labels: labels for the classification of the text
    :param att_vals: attention values (from the annotators)
    :param att_masks: attention masks
    :param params: parameters
    :param is_train: True if the dataset is the training, False otherwise
    :return: DataLoader with the given batch size
    """
    inputs = torch.tensor(input_ids)
    labels = torch.tensor(labels, dtype=torch.long)
    masks = torch.tensor(np.array(att_masks), dtype=torch.uint8)
    attention = torch.tensor(np.array(att_vals), dtype=torch.float)

    data = TensorDataset(inputs, attention, masks, labels)
    if not is_train:
        sampler = SequentialSampler(data)
    else:
        sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=params['batch_size'])
    return dataloader


def set_the_random(seed_val=42):
    """
    Set the random seeds.
    :param seed_val: seed's value
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


def combine_features(tuple_data, params, is_train=False):
    """
    Process the dataset, return a DataLoader.
    :param tuple_data: dataset
    :param params: parameters
    :param is_train: True if the dataset is the training, False otherwise
    :return: DataLoader
    """
    # Get the IDs of the tokens (the corresponding indexes in the vocabulary)
    input_ids = [ele[0] for ele in tuple_data]
    # Get the attention values (from the annotators)
    att_vals = [ele[1] for ele in tuple_data]
    # Get the labels for the classification of the text
    labels = [ele[2] for ele in tuple_data]

    encoder = LabelEncoder()
    if params['num_classes'] == 2:
        encoder.classes_ = np.array(['non-toxic', 'toxic'])
    elif params['num_classes'] == 3:
        encoder.classes_ = np.array(['hatespeech', 'normal', 'offensive'])
    labels = encoder.transform(labels)

    # Pads sequences
    input_ids = pad_sequences(input_ids, maxlen=int(params['max_length']), dtype="long",
                              value=0, truncating="post", padding="post")
    att_vals = pad_sequences(att_vals, maxlen=int(params['max_length']), dtype="float",
                             value=0.0, truncating="post", padding="post")

    # Compute the attention masks
    att_masks = custom_att_masks(input_ids)
    # Compute DataLoader
    dataloader = return_dataloader(input_ids, labels, att_vals, att_masks, params, is_train)
    return dataloader


def load_data_hatexplain(params, tokenizer):
    """
    Load the training, validation and test set.
    :param params: parameters
    :param tokenizer: tokenizer
    :return: training, validation and test set
    """
    # If they don't exist, download the data from HuggingFace
    dataset = load_dataset("hatexplain")
    # Split the dataset
    train = dataset['train']
    val = dataset['validation']
    test = dataset['test']
    # Process the data
    train, val, test = process_data_hatexplain(train, val, test, tokenizer, params)
    return train, val, test


def load_data_sexism_raw(params, tokenizer):
    """
    Load the training set and test set from the unprocessed csv file containing the sexism datas.
    :param params: parameters
    :param tokenizer: tokenizer
    :return: training, validation and test set
    """
    sexism_data = pd.read_csv(params['path_data'] + "/sexism_data.csv")
    sexism_data = indira_preprocess(sexism_data)

    train, test = process_data_distillation(sexism_data, tokenizer, params)

    return train, test


def load_data_sexism(params, tokenizer):
    """
    Load the training set.
    :param params: parameters
    :param tokenizer: tokenizer
    :return: training, validation and test set
    """
    train = load_dataset("json", data_files="Data/sexism_data_preprocessed.json")['train']
    return process_data_sexism(train, tokenizer, params)


def eval_model(params, which_files='test', model=None, test_dataloader=None, device=None):
    """
    Evaluate the model on the training, validation and test set.
    :param params: parameters
    :param which_files: which dataset is being considered
    :param model: model to evaluate
    :param test_dataloader: DataLoader
    :param device: device
    :return: performance metrics
    """
    model.eval()
    print("Running eval on ", which_files, "...")

    true_labels = []
    pred_labels = []
    logits_all = []

    # Evaluate data for one epoch
    for step, batch in tqdm(enumerate(test_dataloader)):
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention vals
        #   [2]: attention mask
        #   [3]: labels
        b_input_ids = batch[0].to(device)
        b_att_val = batch[1].to(device)
        b_input_mask = batch[2].to(device)
        b_labels = batch[3].to(device)

        # Set the model in evaluation phase
        model.eval()

        with torch.no_grad():
            # Compute the output
            outputs = model(b_input_ids, attention_mask=b_input_mask)

        outputs = compute_loss(outputs, params, attention_mask=b_input_mask, attention_vals=b_att_val,
                               labels=b_labels)

        logits = outputs[1]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences and sum
        pred_labels += list(np.argmax(logits, axis=1).flatten())
        true_labels += list(label_ids.flatten())
        logits_all += list(logits)

    logits_all_final = []
    for logits in logits_all:
        logits_all_final.append(softmax(logits))

    # Get the performance metrics
    testf1 = f1_score(true_labels, pred_labels, average='macro')
    testacc = accuracy_score(true_labels, pred_labels)
    testprecision = precision_score(true_labels, pred_labels, average='macro')
    testrecall = recall_score(true_labels, pred_labels, average='macro')

    if params['num_classes'] == 3:
        testrocauc = roc_auc_score(true_labels, logits_all_final, multi_class='ovo', average='macro')
    else:
        testrocauc = 0

    # Report the final accuracy for this validation run.
    print(" Accuracy: {0:.2f}".format(testacc))
    print(" Fscore: {0:.2f}".format(testf1))
    print(" Precision: {0:.2f}".format(testprecision))
    print(" Recall: {0:.2f}".format(testrecall))
    print(" Roc Auc: {0:.2f}".format(testrocauc))

    return testf1, testacc, testprecision, testrecall, testrocauc, logits_all_final


def set_model(params, verbose=True):
    """
    Define the model to be trained.
    :param params: parameter's dictionary
    :param verbose: whether to print the model, default=True
    :return: model to be trained
    """
    model = None
    # Define the model
    if params['model'] == 'bert':
        model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',  # Use the 12-layer BERT model, with an uncased vocab.
            num_labels=params['num_classes'],  # The number of output labels
            output_attentions=True,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
            hidden_dropout_prob=params['dropout_bert']  # The dropout probability for all fully connected layers
        )
    elif params['model'] == 'hate_bert':
        model = BertForSequenceClassification.from_pretrained(
            'GroNLP/hateBERT',  # Use the 12-layer BERT model, with an uncased vocab.
            num_labels=params['num_classes'],  # The number of output labels
            output_attentions=True,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
            hidden_dropout_prob=params['dropout_bert']  # The dropout probability for all fully connected layers
        )
    elif params['model'] == 'hatexplain':
        model = BertForSequenceClassification.from_pretrained(
            'Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two',
            # Use the 12-layer BERT model, with an uncased vocab.
            num_labels=params['num_classes'],  # The number of output labels
            output_attentions=True,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
            hidden_dropout_prob=params['dropout_bert']  # The dropout probability for all fully connected layers
        )
    elif params['model'] == 'bert_mlp':
        model = BertMLP.from_pretrained(
            'bert-base-uncased',  # Use the 12-layer BERT model, with an uncased vocab.
            num_labels=params['num_classes'],  # The number of output labels
            output_attentions=True,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
            hidden_dropout_prob=params['dropout_bert']
        )

    if verbose:
        print(model)
    return model


def train_model(params, train, val, test, model):
    """
    Train the model.
    :param params: parameter's dictionary
    :param train: training set
    :param val: validation set
    :param test: test set
    :param model: model to be trained
    """
    device = params['device']

    # Set the seed value all over the place to make this reproducible
    set_the_random(seed_val=params['random_seed'])

    # Compute weights for each class
    if params['auto_weights']:
        y_train = [ele[2] for ele in train]
        params['weights'] = class_weight.compute_class_weight(class_weight='balanced',
                                                              classes=np.unique(y_train), y=y_train).astype('float32')
    else:
        params['weights'] = [1.0] * params['num_classes']

    # Process the datasets and get the DataLoader
    train_dataloader = combine_features(train, params, is_train=True)
    validation_dataloader = combine_features(val, params, is_train=False)
    test_dataloader = combine_features(test, params, is_train=False)

    # Device
    if params["device"] == 'cuda':
        model.cuda()

    # Set the optimizer
    # args.learning_rate - default is 5e-5, our notebook had 2e-5
    # args.adam_epsilon  - default is 1e-8.
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'], eps=params['epsilon'])

    # Number of training epochs (authors recommend between 2 and 4)
    # Total number of training steps is number of batches * number of epochs
    total_steps = len(train_dataloader) * params['epochs']

    # Create the learning rate scheduler
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps / 10),
                                                             num_training_steps=total_steps)

    # Store the average loss after each epoch, so we can plot them
    loss_values = []

    # Set the values to zero
    best_val_fscore = 0
    best_val_roc_auc = 0
    best_val_precision = 0
    best_val_recall = 0

    for epoch_i in range(0, params['epochs']):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, params['epochs']))
        print('Training...')

        # Reset the total loss for this epoch
        total_loss = 0

        # Set the model in the training mode
        model.train()

        # For each batch of training data
        for step, batch in tqdm(enumerate(train_dataloader)):
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention vals
            #   [2]: attention mask
            #   [3]: labels
            b_input_ids = batch[0].to(device)
            b_att_val = batch[1].to(device)
            b_input_mask = batch[2].to(device)
            b_labels = batch[3].to(device)

            # Set the gradients to zero
            optimizer.zero_grad()

            # Forward pass
            outputs = model(b_input_ids, attention_mask=b_input_mask)
            outputs = compute_loss(outputs, params, attention_mask=b_input_mask, attention_vals=b_att_val,
                                   labels=b_labels)
            loss = outputs[0]

            # Compute the sum of the losses
            # Since `loss` is a Tensor containing a single value,
            # .item()` function just returns the Python value from the tensor
            total_loss += loss.item()
            loss.retain_grad()

            # Perform a backward pass to compute the gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 (to help prevent the "exploding gradients" problem)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient
            optimizer.step()

            # Update the learning rate
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)
        print('avg_train_loss', avg_train_loss)

        # Store the loss value for plotting the learning curve
        loss_values.append(avg_train_loss)

        # Compute the metrix
        train_fscore, train_accuracy, train_precision, train_recall, train_roc_auc, _ = \
            eval_model(params, 'train', model, train_dataloader, device)

        val_fscore, val_accuracy, val_precision, val_recall, val_roc_auc, _ = \
            eval_model(params, 'val', model, validation_dataloader, device)

        # Compute the best scores
        if val_fscore > best_val_fscore:
            print(val_fscore, best_val_fscore)
            best_val_fscore = val_fscore
            best_val_roc_auc = val_roc_auc

            best_val_precision = val_precision
            best_val_recall = val_recall

            if params['to_save']:
                print('Loading BERT tokenizer...')
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False)
                # Save the best model
                save_bert_model(model, tokenizer, params)

    print('best_val_fscore', best_val_fscore)
    print('best_val_rocauc', best_val_roc_auc)
    print('best_val_precision', best_val_precision)
    print('best_val_recall', best_val_recall)

    del model
    torch.cuda.empty_cache()


def eval_model_distillation(which_files='test', model=None, test_dataloader=None, device=None, verbose=True):
    """
    Evaluate the model on the training, validation and test set.
    :param which_files: which dataset is being considered
    :param model: model to evaluate
    :param test_dataloader: DataLoader
    :param device: device
    :param verbose: whether to print the evaluation metrics, default=True
    :return: performance metrics and logits
    """
    if verbose:
        print("Running eval on ", which_files, "...")

    true_labels = []
    pred_labels = []
    logits_all = []

    # Evaluate data for one epoch
    for step, batch in tqdm(enumerate(test_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        # Set the model in evaluation phase
        model.eval()
        # Compute the output
        outputs = model(**batch)
        logits = outputs.logits
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = batch['labels'].to('cpu').numpy()

        # Get predicted labels for this batch
        pred_labels += list(np.argmax(logits, axis=1).flatten())
        true_labels += list(label_ids.flatten())
        logits_all += list(logits)

    logits_all_final = []
    for logits in logits_all:
        logits_all_final.append(softmax(logits))

    # Get the performance metrics
    testf1 = f1_score(true_labels, pred_labels, average='macro')
    testacc = accuracy_score(true_labels, pred_labels)
    testprecision = precision_score(true_labels, pred_labels, average='macro')
    testrecall = recall_score(true_labels, pred_labels, average='macro')

    # Report the final metrics for this evaluation run
    if verbose:
        print(" Accuracy: {0:.4f}".format(testacc))
        print(" Fscore: {0:.4f}".format(testf1))
        print(" Precision: {0:.4f}".format(testprecision))
        print(" Recall: {0:.4f}".format(testrecall))

    return testf1, testacc, testprecision, testrecall, logits_all_final


def train_model_distillation(params, train, test, model, verbose=True):
    """
    Train the model using knowledge distillation on the attention with hateXplain as a teacher model.
    :param params: parameter's dictionary
    :param train: training set
    :param test: test set
    :param verbose: whether to print information on training progress, default=True
    :param model: model to be trained
    """
    device = params['device']

    # get the DataLoaders
    train_dataloader = DataLoader(train, shuffle=True, batch_size=params['batch_size'])
    test_dataloader = DataLoader(test, batch_size=params['batch_size'])

    # Device
    if device == 'cuda':
        model.cuda()

    # Set the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'], eps=params['epsilon'])

    # Total number of training steps is number of batches * number of epochs
    total_steps = len(train_dataloader) * params['epochs']

    # Create the learning rate scheduler
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps / 10),
                                                             num_training_steps=total_steps)

    # Store the average loss after each epoch
    loss_values = []

    # load teacher model according to HateXplain documentation on HuggingFace
    teacher_model = Model_Rational_Label.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two")
    # set it in evaluation mode
    teacher_model.eval()

    for epoch_i in range(0, params['epochs']):
        if verbose:
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, params['epochs']))
            print('Training...')

        # Reset the total loss for this epoch
        total_loss = 0

        # Set the model in the training mode
        model.train()

        # For each batch of training data
        for step, batch in tqdm(enumerate(train_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}

            # clean previously computed gradients
            optimizer.zero_grad()

            # forward pass of both teacher and student model
            outputs_student = model(**batch)
            with torch.no_grad():
                outputs_teacher = teacher_model.to(device)(**batch)

            # compute distillation loss
            loss = compute_distillation_loss(outputs_teacher, outputs_student, params, batch['attention_mask'], batch['labels'])
            total_loss += loss.item()
            loss.retain_grad()

            # Perform a backward pass to compute the gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 (to help prevent the "exploding gradients" problem)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient
            optimizer.step()

            # Update the learning rate
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)
        if verbose:
            print('avg_train_loss', avg_train_loss)

        # Store the loss value
        loss_values.append(avg_train_loss)

        # Compute the metrix
        train_fscore, train_accuracy, train_precision, train_recall, _ = \
            eval_model_distillation('train', model, train_dataloader, device, verbose=verbose)

        test_fscore, test_accuracy, test_precision, test_recall, _ = \
            eval_model_distillation('test', model, test_dataloader, device, verbose=verbose)

    if params['to_save']:
        tokenizer = set_tokenizer(params)
        # save the model after the last epoch
        save_bert_model(model, tokenizer, params)

    if verbose:
        print('final test accuracy', test_accuracy)
        print('final test fscore', test_fscore)
        print('final test precision', test_precision)
        print('final test recall', test_recall)

    del model
    torch.cuda.empty_cache()

    return test_fscore
