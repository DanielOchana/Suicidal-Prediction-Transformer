# Suicidal Prediction Transformer

This repository contains code for a Suicidal Prediction Transformer, which is a Transformer-based model trained from scratch, to predict suicidal tendencies based on textual data.

Project site can be found here:

Project video can be found here: 

The project uses some of the preprocces implemented in :  https://www.kaggle.com/code/bryamblasrimac/suicidal-prediction-multiplemodels-accuracy-73/notebook#5.-Preprocessing 

Data set downloaded from : https://www.kaggle.com/datasets/subhranilpaul/suicidal-mental-health-review

## Results 
### From scratch: 
#### Training accuracy - 77% 
#### Test accuracy - 71.55%
![image](https://github.com/DanielOchana/Suicidal-Prediction-Transformer/assets/102607314/3e308c04-d4ea-4dce-b3ca-e51443278e27)

### GPT2 : 

## Data Preprocessing

The data preprocessing steps include:
1. **removing non-alphabetic**: characters, converting to lowercase, removing stopwords and punctuation, and lemmatizing the remaining tokens

2. **Converting to lowercase**
3. **Removing stopwords and punctuation**: Stopwords are common words that do not contain important meaning and are often removed from texts.
4. **Tokenization**: The text is split into individual words.
5. **Lemmatizing the remaining tokens**: Lemmatization is the process of reducing inflected (or sometimes derived) words to their word stem, base or root form.

6. **Padding**: All text sequences are padded to have the same length. This is necessary because the input to the model needs to be of the same size.

7. **Encoding**: The tokens are converted into numerical values or indices. This is done because the model cannot process raw text and works with numerical data.
8. **Labels mapping to unique IDs** : mapping the labels into o and 1 values 

### Pre-Processed Data 
![image](https://github.com/DanielOchana/Suicidal-Prediction-Transformer/assets/102607314/f7bd433e-4186-40c1-866c-c8bbafa7cccf)


## Model

The model used in this project is a Transformer-based text classifier. It consists of an embedding layer, a positional encoding layer, a Transformer encoder, and a linear layer for classification.

The embedding layer converts the input indices into dense vectors of fixed size. The positional encoding layer adds positional information to the input embeddings. The Transformer encoder processes the input data and the output is passed to the linear layer for classification.

The model is trained using the Adam optimizer and the Binary Cross Entropy with Logits loss function.

## Usage 
### Env
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

### Preprocess Data
A `Data-Pre-Process.ipynb` notebook is provided, that implements all preprocess steps mentioned above. 
The code also saves a preprocess file for later use, so if you have one already (or just want to use the provided csv preprocessed data file, you can skip this step)

The notebook is sequential, so just run all cells in order, and make sure you get a data table with 5 columns at the end: 
text, label, text_prep, token_id, label_prep.

### Train
To use this project, you need to have all the packages above installed. You can then run the `Suicidal-Prediction-From-Scratch.ipynb` notebook to train and evaluate the model.

The code loads the preprocessed file, splits it into train-valid-test sets, and randomizes the hyperparameters. 

Finally, it trains the model, plots the results, and print the accuracy on each of the dataset if needed. 

### Evaluate

10 pretrained models are provided, (`model_checkpoint{i}.pth`).
Each trained model have a corresponding config file (`config{i}.txt` ), with its hyperparams and changes in the architecture. 

A dedicated notebook `From-Scratch-Model-Evaluate.ipynb` is provided, that loads the saved model, and evaluates it on each of the datasets. 

## Folders
* data: original data and preprocessed data CSV files
* code: all provided notebooks. 
* config: config files with the hyperparams used in each model. 
* models: all checkpoint of the trained models. 

## Reference

[1] Document of the spacy model used for preprocces the data - https://spacy.io/models/en  
[2] Previous work on the dataset -  https://www.kaggle.com/code/bryamblasrimac/suicidal-prediction-multiplemodels-accuracy-73/notebook#5.-Preprocessing

[3] Dataset - https://www.kaggle.com/datasets/subhranilpaul/suicidal-mental-health-review


## License ?

This project is open source and available under the [License](LICENSE).

