# Suicidal Prediction Transformer

This repository contains code for a Suicidal Prediction Transformer, which is a Transformer-based model trained from scratch, to predict suicidal tendencies based on textual data. Additionaly, this repository contains code for a pretrained transformer model we finetuned and compared our former model against. Finally, the repository contains records of a session with chatGPT3.5 in which it attempts to predict suicidal tendencies, and a short code analyzing the results of this session.

This work was done as part of the course 046211 Deep Learning at the Technion, course's projects site can be found here: https://taldatech.github.io/ee046211-deep-learning/.

Project video can be found here: https://youtu.be/YIyJAnWMfKE

The project uses some of the preprocces implemented in :  https://www.kaggle.com/code/bryamblasrimac/suicidal-prediction-multiplemodels-accuracy-73/notebook#5.-Preprocessing 

Data set downloaded from : https://www.kaggle.com/datasets/subhranilpaul/suicidal-mental-health-review

## Results 
### From scratch: 
#### Training accuracy - 77% 
#### Test accuracy - 71.55%
![image](https://github.com/DanielOchana/Suicidal-Prediction-Transformer/assets/102607314/3e308c04-d4ea-4dce-b3ca-e51443278e27)

### GPT2 : 
#### Training accuracy - 79% 
#### Test accuracy - 71.8%

### ChatGPT3.5
#### Test accuracy - 75%
![image](https://github.com/DanielOchana/Suicidal-Prediction-Transformer/assets/95130767/2c25af0a-f496-42cf-b66f-50de9c00606f)



## Transformer-based model trained from scratch
### Data Preprocessing

We preprocess the data using the following steps:
1. **removing non-alphabetic**: characters, converting to lowercase, removing stopwords and punctuation, and lemmatizing the remaining tokens

2. **Converting to lowercase**
3. **Removing stopwords and punctuation**: Stopwords are common words that do not contain important meaning and are often removed from texts.
4. **Tokenization**: The text is split into individual words.
5. **Lemmatizing the remaining tokens**: Lemmatization is the process of reducing inflected (or sometimes derived) words to their word stem, base or root form.

6. **Padding**: All text sequences are padded to have the same length. This is necessary because the input to the model needs to be of the same size.

7. **Encoding**: The tokens are converted into numerical values or indices. This is done because the model cannot process raw text and works with numerical data.
8. **Labels mapping to unique IDs** : mapping the labels into o and 1 values 

After preprocessing the data would look like this:
![image](https://github.com/DanielOchana/Suicidal-Prediction-Transformer/assets/102607314/f7bd433e-4186-40c1-866c-c8bbafa7cccf)


### Model and training

The model used in this project is a Transformer-based text classifier. It consists of an embedding layer, a positional encoding layer, a Transformer encoder, and a linear layer for classification.

The embedding layer converts the input indices into dense vectors of fixed size. The positional encoding layer adds positional information to the input embeddings. The Transformer encoder processes the input data and the output is passed to the linear layer for classification.

The model is trained using the Adam optimizer and the Binary Cross Entropy with Logits loss function.

### Usage 
#### Env
The repository provides an environment.yml file.
This file contains a list of all the packages needed for using the code. 
To create a Conda environment and install all packages run:

``` bash 
conda env create -f environment.yml
```
#### Main Packages Versions
torch==2.2.2 

pandas==2.0.3

numpy==1.24.4

transformers==4.38.2

spacy==3.7.4

#### Preprocess Data
A `Data-Pre-Process.ipynb` notebook is provided, that implements all preprocess steps mentioned above. 
The code also saves a preprocess file for later use, so if you have one already (or just want to use the provided csv preprocessed data file, you can skip this step)

The notebook is sequential, so just run all cells in order, and make sure you get a data table with 5 columns at the end: 
text, label, text_prep, token_id, label_prep.

#### Train
To use this project, you need to have all the packages above installed. You can then run the `Suicidal-Prediction-From-Scratch.ipynb` notebook to train and evaluate the model.

The code loads the preprocessed file, splits it into train-valid-test sets, and randomizes the hyperparameters. 

Finally, it trains the model, plots the results, and print the accuracy on each of the dataset if needed. 

#### Evaluate

10 pretrained models are provided, (`model_checkpoint{i}.pth`).
Each trained model have a corresponding config file (`config{i}.txt` ), with its hyperparams and changes in the architecture. 

A dedicated notebook `From-Scratch-Model-Evaluate.ipynb` is provided, that loads the saved model, and evaluates it on each of the datasets. 

## Pretrained model
As mentioned in the introduction, we also finetuned a pretrained model to create a baseline to compare the previous model against. The pretrained model we worked with was GPT2ForSequenceClassification model with the standard configuration GPT2Config.from_pretrained() and the standard tokenizer GPT2Tokenizer.from_pretrained(). For the pretrained model, we used a lighter preprocessing scheme, only tokenizing the data and mapping the labels to $0, 1$. For training, we used the AdamW optimizer, with a linearly decreasing scheduler and the cross entropy loss. The training consisted of two main parts. In the first part, we fine-tuned all of the weights of the model. In the second part, we only updated the final layer of the model. 

The code preprocesses the dataset and splits it in to train and test, loads the pretrained model, finetunes it  according to the hyperparameters in the beginning of the notebook, and evaluates the finetuned model on the test set.

## ChatGPT3.5 baseline
Using the standard chat interface (not using the API), we first explained to ChatGPT that we want it to help us classify for each text segment we gave it, whether that person should be put on suicide watch or if he just has depression. In this explanation we also provided him with 2 real labeled examples from our dataset (one with depression label and the other with suicide watch label). Afterwards we queried it on a single text entry at a time, and asked him to share his prediction. In total, we queried ChatGPT on $100$ texts from the dataset, with $50$ labeled depression on the other $50$ labeled as suicide watch. Note that due to the sensitive nature of mental health topics, ChatGPT has very strict usage policies which made this experiment rather difficult which effects the results obtained. The code provided analyses ChatGPT answers documented in a excel file.

## Folders
* Under "from scratch" is all the relevant code and data for the mdoel we trained from scratch:
  * data: original data and preprocessed data CSV files
  * code: all provided notebooks. 
  * config: config files with the hyperparams used in each model. 
  * models: all checkpoint of the trained models. 
* "Pretrained" folder contains the notebook we used to finetune the pretrained model, and a file containing links to a checkpoint of a finetuned model.
* "chat GPT" folder contains a session with ChatGPT3.5 in which it attempts to classify instances from the same dataset, a table documenting it's answers and a short script analysing it's responses. 

## Reference

[1] Document of the spacy model used for preprocces the data - https://spacy.io/models/en  
[2] Previous work on the dataset, including preprocessing -  https://www.kaggle.com/code/bryamblasrimac/suicidal-prediction-multiplemodels-accuracy-73/notebook#5.

[3] Dataset - https://www.kaggle.com/datasets/subhranilpaul/suicidal-mental-health-review


## License

This project is licensed under the MIT License. For more details, see the [MIT License documentation](https://opensource.org/licenses/MIT).


