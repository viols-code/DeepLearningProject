### Parameters used in this code
This document notes down all the hyper-parameters associated with data, models and evaluation. Before adding any new options to some of the categories, please ensure that is implemented in the respective python file.
Parameters other than the ones described below are to be kept fixed.

#### Attention aggregation parameters
* **type_attention** : How the normalisation of the attention vector will happen. Three options are available currently "softmax","neg_softmax" and "sigmoid". More details [here](https://github.com/punyajoy/HateXplain/blob/master/Preprocess/attentionCal.py).
* **variance**: constant multiplied with the attention vector to increase the difference between the attention to attended and non-attended tokens.More details [here](https://github.com/punyajoy/HateXplain/blob/master/Preprocess/attentionCal.py). 
* **aggr_attention**: This can be set as *'mean'*, *'lenient'* or 'conservative'. Mean perform the mean of the rationales from the annotators, lenient perform an OR operation and conservative perform an AND operation.
* **number_rationales**: Set the number of rationales to have for each sample.
* **evaluate**: To evaluate the different types of attentions.

~~~
variance=5
attention = [0,0,0,1,0]
attention_modified = attention * variance
attention_modified = [0,0,0,5,0]
~~~

#### Decay parameters
* **alpha**: Computes the reserve value for the distribute function.
* **method**: It can be 'additive' or 'geometric'. If set to additive, a uniform distribution is used on the reserve value. If set to geometric, a geometric distribution is used.
* **normalized**: Normalize the final attention weights to make them sum up to one.
* **p_value**: p-value used for the geometric distribution.
* **window**: Window to be considered for the context when performing the decay.
* **decay**: If true, decay is performed.

#### Preprocessing parameters
* **include_special** : This can be set as *True* or *False*. This is with respect to [ekaphrasis](https://github.com/cbaziotis/ekphrasis) processing. For example ekphrasis adds a special character to the hashtags after processing. for e.g. #happyholidays will be transformed to <hashtag> happy holidays </hashtag>. If set the `include specials` to false than the begin token and ending token will be removed. 
* **max_length** : This represents the maximum length of the words in case of non transformers models or subwords in case of transformers models. For all our models this is set to 128.
* **majority**: This is the number of annotators that need to agree in order to not have indecision. Default is 2.
* **tokenizer**: either 'default' or 'bert_miniature'. If default, will use the default tokenizer corresponding to the model specified in "model". Else will use the bert miniature tokenizer.

##### Common parameters for training 
* **att_lambda**: Contribution of the attention loss to the total loss.
* **auto_weights**: This can be set as *True* or *False*. True will assign the class weight based on the class distribution in the training dataset.
* **batch_size**: Batch size to train the model. We set it to 32 for every model.
* **device**: Device on which the system will run. set "cuda(gpu)" or "cpu".
* **distillation_alpha**: When training with the distillation loss, indicates the weight of the attention loss wrt the labels loss
* **epochs**: Number of epochs to train the model.
* **epsilon**: Used as a parameter in Adam optimizer. Default set at 1e-08.
* **learning_rate**: Learning rate passed to the Adam optimizer. For BERT, it is set to closer to 2e-5, for non-transformer model it is in the range of 0.001 to 0.1 in our case. 
* **model**: Model to be fine-tuned. Possible options are: 'bert', 'hate_bert' and 'hatexplain'.
* **path_files** : Path where to save the best model.
* **weights**: If you want to manually set the weights for the different classes. Be sure to maintain a vector of length similar to the original number of classes. 
* **train_att**: This can be set as *True* or *False*. This will be set as True if you want to train the attention weights of the model.
* **dropout_bert**: Dropout after the linear layer in BERT.
* **num_supervised_heads**: Number of attention heads whose final attention needs to be aligned with the ground truth attention,
* **save_only_bert**: This can be set as *True* or *False*. "True" will save the BERT part of the model only not the linear layer.
* **supervised_layer_pos**: The layer whose attention heads needs to be aligned in the final layer.
* **num_classes**: These represent the number of classes. It could be `3` or `2`.
* **random_seed**: This should be set to `42` for reproducibility.
* **to_save**: This can be set as *True* or *False*. It controls if you want to save the final model or not.