# Training and Inference

All the complexity is contained in the wrapper.py file. We have decided to do so, since there were different parameters to take into consideration, and we thought this would have been the best way to not make mistake when training different models with different datasets.
The idea is to just change the function of the wrapper called in main.py.
Here we present a list of the functions and what they do.
The only thing to take into account is the device, that needs to be set correctly before starting the training and the inference.

## Requirements
Install the libraries using the following command (preferably inside an environment)
~~~
pip install -r requirements.txt
~~~

## Datasets

Notice that for HateXplain, everything is automatically downloaded from Hugging Face.
For sexism dataset it is required to download data from [here](https://doi.org/10.7802/2251) and unzip it in the data folder and call them sexism_data.csv.
To use sexism dataset in aggregation to HateXplain dataset, you first need to call sexism_preprocessing.py which will create a sexism_data_preprocessed.json.
~~~
python preprocessing_sexism_dataset/sexism_preprocessing.py                    
~~~

## Training

### Train different models:
- train_bert_basic: Train BERT model on HateXplain dataset.
- train_hate_bert_basic: Train HateBERT model on HateXplain dataset.
- train_bert_mlp: Train BERT + MLP model on HateXplain dataset.
- fine_tune_hatexplain_model: Fine tune HateXplain model on sexism dataset.
- train_bert_aggregated_dataset: Train BERT model on HateXplain dataset aggregated with sexism dataset.
- train_hate_bert_aggregated_dataset: Train HateBERT model on HateXplain dataset aggregated with sexism dataset.
- fine_tune_bert_sexism_dataset: Fine-tune BERT model on sexism dataset to replicate the results from Samory et al. 2021.
- fine_tune_bert_distillation: Fine-tune BERT model on sexism dataset using knowledge distillation with HateXplain as a teacher model.

### Examples:
Only the main.py file needs to be changed for the training. Here we show two examples.
If we want to train BERT model on HateXplain dataset:
~~~
from wrapper import Wrapper

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    wrapper = Wrapper()
    wrapper.fine_tune_bert_sexism_dataset()
~~~

If we want to train BERT model on HateXplain dataset aggregated with sexism dataset:
~~~
from wrapper import Wrapper

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    wrapper = Wrapper()
    wrapper.train_bert_aggregated_dataset()
~~~
The only difference is the function called by the wrapper.
Once the main.py file has been modified to call for the wanted function of the wrapper, you can call:
~~~
python main.py                           
~~~

## Inference

Test the model:
- compute_bias: Compute bias. Set the correct number of classes and the path to the model.
- compute_explainability: Computes explainability metrics. Set the correct number of classes and the path to the model.
Notice: The bias and explainability can be computed only for hateXplain dataset, and not for sexism dataset.
This is because for the bias in sexism dataset the only community would be Women, and for the explainability we would have to reduce the dataset to only posts that have adversarial examples.

Example:

Modification of the main.py file:
~~~
from wrapper import Wrapper

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    wrapper = Wrapper()
    wrapper.compute_bias()
~~~

Once the main.py file has been modified to call for the wanted function of the wrapper, you can call:
~~~
python main.py                           
~~~

