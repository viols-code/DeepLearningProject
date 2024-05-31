# Improving explainability of sexism detection in social media text

**Group**: Corentin Genton, Lucille Niederhauser, Viola Renne  (Group 11)  
**Course**: EE-559 - Deep Learning   
**Teacher**: Cavallaro Andrea   

***WARNING: The repository contains content that are offensive and/or hateful in nature.***
 
***REFERENCES:***
In this repository, we replicate and build upon the results of the following papers:
1) [HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection](https://arxiv.org/abs/2012.10289). [HateXplain GitHub repository](https://github.com/hate-alert/HateXplain)
2) [Exploring Hate Speech Detection with HateXplain and BERT](https://arxiv.org/abs/2208.04489). [GitHub repository](https://github.com/sayani-kundu/11711-HateXplain)
3) [“Call me sexist, but...”: Revisiting Sexism Detection Using Psychological Scales
and Adversarial Samples](https://arxiv.org/pdf/2004.12764.pdf). [GitHub repository](https://github.com/gesiscss/theory-driven-sexism-detection)
4) [Social Media Hate Speech Detection Using Explainable Artificial Intelligence (XAI)](https://www.mdpi.com/1999-4893/15/8/291).

These models are used in the code:
1) [BERT base model (uncased)](https://huggingface.co/google-bert/bert-base-uncased)
2) [HateXplain - two classes](https://huggingface.co/Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two)
3) [HateBERT](https://huggingface.co/GroNLP/hateBERT)


### Abstract
Many deep-learning models have been developed to detect hate speech in social media. Though they can achieve good performances, their decision-making processes are often unclear, making it harder for models to comply with certain ethical and legal guidelines. We thus decided to study how models for hate speech detection, and more precisely sexism detection, could gain in explainability. We explored methods building on the work of Mathew et al. 2022 mainly through different pre-processing, fine-tuning of existing models and knowledge distillation.

------------------------------------------
***Folder Description***:
------------------------------------------
~~~

./Data                         --> Contains the dataset related files.
./preprocessing_sexism_dataset --> Contains the code for processing sexism dataset
./Pretrained_Model             --> Contains the definition of BERT + MLP model
./Rationales                   --> Contains the figures for the rationales analysis

~~~

Notice that for HateXplain, everything is automatically downloaded from Hugging Face.
For sexism dataset it is required to download data from [here](https://doi.org/10.7802/2251) and unzip it in the data folder and call them sexism_data.csv.
To use sexism dataset in aggregation to HateXplain dataset, you first need to call sexism_preprocessing.py which will create a sexism_data_preprocessed.json.
~~~
python preprocessing_sexism_dataset/sexism_preprocessing.py                    
~~~

------------------------------------------
***Table of contents***:
------------------------------------------

[**Parameters**](Parameters_description.md) : This describes all the different parameter that are used in this code

------------------------------------------
***Usage instructions*** 
------------------------------------------
Install the libraries using the following command (preferably inside an environment)
~~~
pip install -r requirements.txt
~~~

#### Training
To train the basic model use the following command.
~~~
python main.py                           
~~~

Everything is handled by the wrapper, to limit the complexity.
[**Training and Inference**](Training_and_Inference.md) : This describes how to train and use our models.

### Models
[Google drive link for dowloadable models](https://drive.google.com/drive/folders/1qx4Lfo-JRkxqOF-x3gjM5YiTLZkLIYnG?usp=sharing)