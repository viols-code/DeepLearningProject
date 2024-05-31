from train import *
from data_processing import *
from sklearn.model_selection import KFold
from compute_bias import get_bias_dict
from utils import *
from utils_explainability import *
from compute_explainability import save_explainability


class Wrapper:
    def __init__(self):
        self.params = {
            'alpha': 0.5,
            'att_lambda': 0.001,
            'auto_weights': True,
            'batch_size': 16,
            'device': 'cpu',
            'distillation_alpha': 0.079,
            'dropout_bert': 0.1,
            'epochs': 5,
            'epsilon': 1e-08,
            'include_special': False,
            'learning_rate': 2e-05,
            'majority': 2,
            'max_length': 128,
            'method': 'additive',
            'model': 'bert',
            'normalized': False,
            'num_classes': 2,
            'num_supervised_heads': 6,
            'p_value': 0.8,
            'path_data': 'Data',
            'path_files': 'bert-base-uncased',
            'random_seed': 42,
            'supervised_layer_pos': 11,
            'save_data': False,
            'tokenizer': 'default',
            'to_save': True,
            'train_att': True,
            'type_attention': 'softmax',
            'aggr_attention': 'mean',
            'number_rationales': 3,
            'variance': 5,
            'window': 4.0,
            'decay': False,
            'evaluate': False
        }

    def set_device(self, device):
        """
        Set device.
        :param device: device, it could be 'cpu' or 'cuda'
        """
        self.params['device'] = device

    def set_model(self, model):
        """
        Set model.
        :param model: model
        """
        self.params['model'] = model

    def set_number_rationales(self, number_rationales):
        """
        Set number rationales.
        :param number_rationales: number rationales
        """
        self.params['number_rationales'] = number_rationales

    def set_epochs(self, epochs):
        """
        Set epochs.
        :param epochs: number of epochs
        """
        self.params['epochs'] = epochs

    def set_evaluate(self, evaluate):
        """
        Set evaluate.
        :param evaluate: evaluate
        """
        self.params['evaluate'] = evaluate

    def set_aggr_attention(self, aggr_attention):
        """
        Set aggregate attention.
        :param aggr_attention: type of aggregation used for the attention
        """
        self.params['aggr_attention'] = aggr_attention

    def set_decay(self, decay):
        """
        Set decay.
        :param decay: True to use decay, false otherwise
        """
        self.params['decay'] = decay

    def set_window(self, window):
        """
        Set window.
        :param window: number of tokens considered to left and right.
        """
        self.params['window'] = window

    def set_method(self, method):
        """
        Set method.
        :param method: method to use
        """
        self.params['method'] = method

    def set_normalized(self, normalized):
        """
        Set normalized.
        :param normalized: True to normalize the data
        """
        self.params['normalized'] = normalized

    def train_bert_basic(self):
        """
        Train BERT model on HateXplain dataset.
        """
        self.set_device('cuda')
        self.set_model('bert')
        # Define the model
        model = set_model(self.params)
        # Define the tokenizer
        tokenizer = set_tokenizer(params=self.params)
        # Load the data
        train, val, test = load_data_hatexplain(self.params, tokenizer)
        # Train the model
        train_model(self.params, train, val, test, model)

    def train_hate_bert_basic(self):
        """
        Train HateBERT model on HateXplain dataset.
        """
        self.set_device('cuda')
        self.set_model('hate_bert')
        # Define the model
        model = set_model(self.params)
        # Define the tokenizer
        tokenizer = set_tokenizer(params=self.params)
        # Load the data
        train, val, test = load_data_hatexplain(self.params, tokenizer)
        # Train the model
        train_model(self.params, train, val, test, model)

    def train_bert_mlp(self):
        """
        Train BERT + MLP model on HateXplain dataset.
        """
        self.set_device('cuda')
        self.set_model('bert_mlp')
        self.set_epochs(50)
        # Define the model
        model = set_model(self.params)
        # Define the tokenizer
        tokenizer = set_tokenizer(params=self.params)
        # Load the data
        train, val, test = load_data_hatexplain(self.params, tokenizer)
        # Train the model
        train_model(self.params, train, val, test, model)

    def fine_tune_hatexplain_model(self):
        """
        Fine tune HateXplain model on sexism dataset.
        """
        self.set_device('cuda')
        self.set_model('hatexplain')
        # Define the model
        model = set_model(self.params)
        # Define the tokenizer
        tokenizer = set_tokenizer(params=self.params)
        # Load the dataset
        self.set_number_rationales(1)
        sexism_data = load_data_sexism(self.params, tokenizer)
        # Set a seed for reproducibility
        np.random.seed(42)

        # Shuffle the data
        np.random.shuffle(sexism_data)

        # Calculate the lengths of train, val, and test sets
        total_length = len(sexism_data)
        train_length = int(total_length * 0.8)
        val_length = int(total_length * 0.1)

        # Split the data
        train = sexism_data[:train_length]
        val = sexism_data[train_length:train_length + val_length]
        test = sexism_data[train_length + val_length:]
        # Train the model
        self.set_number_rationales(3)
        train_model(self.params, train, val, test, model)

    def train_bert_aggregated_dataset(self):
        """
        Train BERT model on HateXplain dataset aggregated with sexism dataset.
        """
        self.set_device('cuda')
        self.set_model('bert')
        # Define the model
        model = set_model(self.params)
        # Define the tokenizer
        tokenizer = set_tokenizer(params=self.params)
        # Load the two datasets
        train, val, test = load_data_hatexplain(self.params, tokenizer)
        # Change the number of rationales
        self.set_number_rationales(1)
        sexism_data = load_data_sexism(self.params, tokenizer)
        self.set_number_rationales(3)
        # Aggregate the data
        train = train + sexism_data
        # Train
        train_model(self.params, train, val, test, model)

    def train_hate_bert_aggregated_dataset(self):
        """
        Train HateBERT model on HateXplain dataset aggregated with sexism dataset.
        """
        self.set_device('cuda')
        self.set_model('hate_bert')
        # Define the model
        model = set_model(self.params)
        # Define the tokenizer
        tokenizer = set_tokenizer(params=self.params)
        # Load the two datasets
        train, val, test = load_data_hatexplain(self.params, tokenizer)
        # Change the number of rationales
        self.set_number_rationales(1)
        sexism_data = load_data_sexism(self.params, tokenizer)
        self.set_number_rationales(3)
        # Aggregate the data
        train = train + sexism_data
        # Train
        train_model(self.params, train, val, test, model)

    def evaluate_attention_before_softmax(self):
        """
        Plots graph to see the number of zeros in the rationales.
        """
        # Set parameters for the analysis
        self.set_device('cpu')
        self.set_model('bert')
        self.set_evaluate(True)

        # Get data
        tokenizer = set_tokenizer(params=self.params)
        train = load_dataset("hatexplain", split="train")

        # Mean aggregation
        train1 = collect_data(train, tokenizer, params=self.params)
        train1 = encode_data(train1)
        attentions1 = [np.sum(post[1] == 0) / len(post[1]) for post in train1 if post[2] not in ['normal', 'non-toxic']]
        # Lenient (OR) aggregation
        self.set_aggr_attention('lenient')
        train2 = collect_data(train, tokenizer, params=self.params)
        train2 = encode_data(train2)
        attentions2 = [np.sum(post[1] == 0) / len(post[1]) for post in train2 if post[2] not in ['normal', 'non-toxic']]
        # Conservative (AND) aggregation
        self.set_aggr_attention('conservative')
        train3 = collect_data(train, tokenizer, params=self.params)
        train3 = encode_data(train3)
        attentions3 = [np.sum(post[1] == 0) / len(post[1]) for post in train3 if post[2] not in ['normal', 'non-toxic']]

        save_images_zeros(attentions1, attentions2, attentions3, text='hatexplain')

        # Set parameters
        self.set_number_rationales(1)
        self.set_aggr_attention('mean')

        # Mean aggregation
        train1 = load_data_sexism(self.params, tokenizer)
        attentions1 = [np.sum(post[1] == 0) / len(post[1]) for post in train1 if post[2] == 'toxic']
        # Lenient (OR) aggregation
        self.set_aggr_attention('lenient')
        train2 = load_data_sexism(self.params, tokenizer)
        attentions2 = [np.sum(post[1] == 0) / len(post[1]) for post in train2 if post[2] == 'toxic']
        # Conservative (AND) aggregation
        self.set_aggr_attention('conservative')
        train3 = load_data_sexism(self.params, tokenizer)
        attentions3 = [np.sum(post[1] == 0) / len(post[1]) for post in train3 if post[2] == 'toxic']

        save_images_zeros(attentions1, attentions2, attentions3, text='sexism')

    def evaluate_attention_hatexplain_after_softmax(self):
        """
        Evaluate attention values for HateXplain for different types of aggregation.
        """
        # Set parameters
        self.set_device('cpu')
        self.set_model('bert')

        tokenizer = set_tokenizer(params=self.params)
        # Get data
        train = load_dataset("hatexplain", split="train")

        # Mean aggregation
        train1 = collect_data(train, tokenizer, params=self.params)
        train1 = encode_data(train1)
        # Lenient (OR) aggregation
        self.set_aggr_attention('lenient')
        train2 = collect_data(train, tokenizer, params=self.params)
        train2 = encode_data(train2)
        # Conservative (AND) aggregation
        self.set_aggr_attention('conservative')
        train3 = collect_data(train, tokenizer, params=self.params)
        train3 = encode_data(train3)

        save_images_rationales(train1, train2, train3, tokenizer, text="", dir='HateXplain')


    def evaluate_attention_sexism_after_softmax(self):
        """
        Evaluate attention values for Sexism dataset for different types of aggregation.
        """
        # Set parameters
        self.set_device('cpu')
        self.set_model('bert')
        self.set_number_rationales(1)

        tokenizer = set_tokenizer(params=self.params)
        # Get data for mean aggregation
        train1 = load_data_sexism(self.params, tokenizer)
        # Lenient (OR) aggregation
        self.set_aggr_attention('lenient')
        train2 = load_data_sexism(self.params, tokenizer)
        # Conservative (AND) aggregation
        self.set_aggr_attention('conservative')
        train3 = load_data_sexism(self.params, tokenizer)

        save_images_rationales(train1, train2, train3, tokenizer, text="", dir='Sexism')


    def evaluate_attention_sexism_after_softmax_decay_additive(self):
        """
        Evaluate attention values for Sexism dataset for different types of aggregation and decay.
        """
        # Set parameters
        self.set_device('cpu')
        self.set_model('bert')
        self.set_number_rationales(1)
        self.set_decay(True)
        self.set_window(2)
        self.set_method('additive')
        self.set_normalized(True)

        tokenizer = set_tokenizer(params=self.params)
        # Get data for mean aggregation
        train1 = load_data_sexism(self.params, tokenizer)
        # Lenient (OR) aggregation
        self.set_aggr_attention('lenient')
        train2 = load_data_sexism(self.params, tokenizer)
        # Conservative (AND) aggregation
        self.set_aggr_attention('conservative')
        train3 = load_data_sexism(self.params, tokenizer)

        save_images_rationales(train1, train2, train3, tokenizer, text="_decay_additive", dir='Sexism')


    def evaluate_attention_sexism_after_softmax_decay_geometric(self):
        """
        Evaluate attention values for Sexism dataset for different types of aggregation and decay.
        """
        # Set parameters
        self.set_device('cpu')
        self.set_model('bert')
        self.set_number_rationales(1)
        self.set_decay(True)
        self.set_window(2)
        self.set_method('geometric')
        self.set_normalized(True)

        tokenizer = set_tokenizer(params=self.params)
        # Get data for mean aggregation
        train1 = load_data_sexism(self.params, tokenizer)
        # Lenient (OR) aggregation
        self.set_aggr_attention('lenient')
        train2 = load_data_sexism(self.params, tokenizer)
        # Conservative (AND) aggregation
        self.set_aggr_attention('conservative')
        train3 = load_data_sexism(self.params, tokenizer)

        save_images_rationales(train1, train2, train3, tokenizer, text="_decay_geometric", dir='Sexism')


    def compute_bias(self):
        """
        Compute bias.
        """
        self.set_device('cpu')
        # REMEMBER TO SET THE RIGHT NUMBER OF CLASSES
        self.params['num_classes'] = 2
        test = load_dataset("hatexplain", split="test")
        model_dir = "Trained_Model/BERT-2"

        model = BertForSequenceClassification.from_pretrained(model_dir)
        tokenizer = BertTokenizer.from_pretrained(model_dir)

        test_data = collect_data(test, tokenizer, self.params)
        test_data = encode_data(test_data)
        test_dataloader = combine_features(test_data, self.params, is_train=False)
        test = get_annotated_data(test, params=self.params)

        model.eval()
        true_labels = []
        pred_labels = []
        logits_all = []

        with torch.no_grad():
            for step, batch in tqdm(enumerate(test_dataloader)):
                b_input_ids = batch[0].to(self.params['device'])
                b_att_val = batch[1].to(self.params['device'])
                b_input_mask = batch[2].to(self.params['device'])
                b_labels = batch[3].to(self.params['device'])

                # Compute the output
                outputs = model(b_input_ids, attention_mask=b_input_mask)
                logits = outputs[0]
                # Calculate the accuracy for this batch of test sentences and sum
                pred_labels += list(np.argmax(logits, axis=1).flatten())
                true_labels += list(b_labels.flatten())
                logits_all += list(logits)

            logits_all_final = []
            probabilities = []
            for logits in logits_all:
                probability = softmax(logits.numpy())
                if self.params['num_classes'] == 2:
                    dictionary = {
                        'non-toxic': probability[0],
                        'toxic': probability[1]
                    }
                else:
                    dictionary = {
                        'non-toxic': probability[1],
                        'toxic': probability[0] + probability[2]
                    }
                logits_all_final.append(dictionary)
                probabilities.append(probability)

            # Get the performance metrics
            test_f1 = f1_score(true_labels, pred_labels, average='macro')
            print()
            test_acc = accuracy_score(true_labels, pred_labels)
            test_precision = precision_score(true_labels, pred_labels, average='macro')
            test_recall = recall_score(true_labels, pred_labels, average='macro')

            if self.params['num_classes'] == 3:
                testr_ocauc = roc_auc_score(true_labels, probabilities, multi_class='ovo', average='macro')
            else:
                testr_ocauc = 0

            # Report the final accuracy for this validation run.
            print(" Accuracy: {0:.4f}".format(test_acc))
            print(" Fscore: {0:.4f}".format(test_f1))
            print(" Precision: {0:.4f}".format(test_precision))
            print(" Recall: {0:.4f}".format(test_recall))
            print(" Roc Auc: {0:.4f}".format(testr_ocauc))

        target_group_list = ['African', 'Islam', 'Jewish', 'Homosexual', 'Women', 'Refugee', 'Arab', 'Caucasian',
                             'Asian', 'Hispanic']
        method_list = ['subgroup', 'bpsn', 'bnsp']
        get_bias_dict(method_list, target_group_list, test, logits_all_final)

    def compute_explainability(self):
        """
        Computes explainability metrics.
        """
        self.set_device('cpu')
        # REMEMBER TO SET THE RIGHT NUMBER OF CLASSES
        self.params['num_classes'] = 2
        test = load_dataset("hatexplain", split="test")
        model_dir = "Trained_Model/BERT-2"

        model = BertForSequenceClassification.from_pretrained(model_dir)
        tokenizer = BertTokenizer.from_pretrained(model_dir)
        save_explainability(self.params, test, model, tokenizer)

        data_all_labelled = get_annotated_data(test, self.params)
        training_data = get_training_data(data_all_labelled, self.params, tokenizer)
        method = 'union'
        save_split = True
        save_path = './metrics/'  # The dataset in Eraser Format will be stored here.
        convert_to_eraser_format(training_data, method, save_split, save_path, 'test')

        result_file_dir = '../explanations_dicts/bert_0.001_explanation_top5.json'
        score_file_dir = '../model_explain_output.json'
        os.chdir('./eraserbenchmark')
        new_save_path = '../metrics/'
        os.system(
            'PYTHONPATH=./:$PYTHONPATH python rationale_benchmark/metrics.py --split test --strict --data_dir {} --results {} --score_file {} --num_classes {}'.format(
                new_save_path, result_file_dir, score_file_dir, self.params['num_classes']))

    def fine_tune_bert_sexism_dataset(self, num_iter=5):
        """
        Fine-tune BERT model on sexism dataset to replicate the results from Samory et al. 2021.
        """
        self.set_device('cuda')
        self.set_model('bert')
        self.params['distillation_alpha'] = 0
        f_scores = []
        # training and evaluation is performed multiple times on data split differently each time
        for i in range(num_iter):
            # define the model
            model = set_model(self.params)
            # define the tokenizer
            tokenizer = set_tokenizer(params=self.params)
            # load and split the data
            train, test = load_data_sexism_raw(self.params, tokenizer)
            # train
            f_score = train_model_distillation(self.params, train, test, model)
            f_scores.append(f_score)

        print(f'average F1-score over 5 runs: {np.mean(f_scores)}, std: {np.std(f_scores)}')

    def fine_tune_bert_distillation(self):
        """
        Fine-tune BERT model on sexism dataset using knowledge distillation with HateXplain as a teacher model.
        """
        self.set_device('cuda')
        self.set_model('bert')
        # set the seed
        set_the_random(seed_val=self.params['random_seed'])
        # define the model
        model = set_model(self.params)
        # define the tokenizer
        tokenizer = set_tokenizer(params=self.params)
        # load and split the data
        train, test = load_data_sexism_raw(self.params, tokenizer)
        # train
        f_score = train_model_distillation(self.params, train, test, model)

    def cross_validate_distillation(self, num_folds=5, alpha_values=np.linspace(0.001, 0.5, 20, endpoint=True)):
        """
        Cross-validation on fine-tuned bert model on sexism dataset to find the optimal distillation alpha.
        """
        # set the seed
        set_the_random(seed_val=self.params['random_seed'])

        # don't save the CV models
        self.params['to_save'] = False

        # define the tokenizer
        tokenizer = set_tokenizer(params=self.params)

        # load and split data, cross-validation only performed on train set
        train, _ = load_data_sexism_raw(self.params, tokenizer)

        # get the folds
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=self.params['random_seed'])
        results = {alpha: [] for alpha in alpha_values}

        for alpha in alpha_values:
            print(f"###### Starting cross-validation with alpha: {alpha} ######")
            # set the alpha
            self.params['distillation_alpha'] = alpha

            # cross-validation
            for train_idx, val_idx in kf.split(train):
                model = set_model(self.params, verbose=False)
                train_data = [train[int(i)] for i in train_idx]
                val_data = [train[int(i)] for i in val_idx]
                f_score = train_model_distillation(self.params, train_data, val_data, model, verbose=False)
                results[alpha].append(f_score)

            print(f"###### Finished cross-validation for alpha {alpha} ######")
            print(f"###### Mean F-score {np.mean(results[alpha])} ######")

        # compute mean and std deviation for each alpha
        mean_scores = {alpha: np.mean(scores) for alpha, scores in results.items()}
        std_scores = {alpha: np.std(scores) for alpha, scores in results.items()}
        print("Mean F-scores:", mean_scores)
        print("Standard Deviation of F-scores:", std_scores)

        # get best alpha
        best_alpha = max(mean_scores, key=mean_scores.get)
        print(f"Best alpha: {best_alpha}, Mean F1-score: {mean_scores[best_alpha]}")
