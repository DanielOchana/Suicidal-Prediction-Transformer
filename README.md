# Suicidal Prediction Transformer

This repository contains code for a Suicidal Prediction Transformer, which is a Transformer-based model trained from scratch, to predict suicidal tendencies based on textual data.

Project site can be found here:

Project video can be found here: 

The project uses some of the preprocces implemented in :  https://www.kaggle.com/code/bryamblasrimac/suicidal-prediction-multiplemodels-accuracy-73/notebook#5.-Preprocessing 

Data set downloaded from : https://www.kaggle.com/datasets/subhranilpaul/suicidal-mental-health-review


## Data Preprocessing

The data preprocessing steps include:
1. **removing non-alphabetic**: characters, converting to lowercase, removing stopwords and punctuation, and lemmatizing the remaining tokens

2. **Converting to lowercase**
3. **Removing stopwords and punctuation**: Stopwords are common words that do not contain important meaning and are often removed from texts.
4. **Tokenization**: The text is split into individual words.
5. **Lemmatizing the remaining tokens**: Lemmatization is the process of reducing inflected (or sometimes derived) words to their word stem, base or root form.

6. **Padding**: All text sequences are padded to have the same length. This is necessary because the input to the model needs to be of the same size.

7. **Encoding**: The tokens are converted into numerical values or indices. This is done because the model cannot process raw text and works with numerical data.

### Pre Proccessed Data 
![image](https://github.com/DanielOchana/Suicidal-Prediction-Transformer/assets/102607314/f7bd433e-4186-40c1-866c-c8bbafa7cccf)


## Model

The model used in this project is a Transformer-based text classifier. It consists of an embedding layer, a positional encoding layer, a Transformer encoder, and a linear layer for classification.

The embedding layer converts the input indices into dense vectors of fixed size. The positional encoding layer adds positional information to the input embeddings. The Transformer encoder processes the input data and the output is passed to the linear layer for classification.

The model is trained using the Adam optimizer and the Binary Cross Entropy with Logits loss function.

## Usage 
### Env
The repository provides an environment.yml file.
This file contains a list of all the packages needed for using the code. 
In order to create a Conda environment and install all packages run:

``` bash 
conda env create -f environment.yml
```

### Train
To use this project, you need to have Python and PyTorch installed. You can then run the `deep_learning_project.ipynb` notebook to train and evaluate the model.

After preprocces once, the code saves the preproccesed data for later use. 
Just skip the preprocces section and load the saved preproccesed data. 

The repository provides a preproccesed data file. 

### Evaluate

10 pretrained models are provided, ("model_checkpoint_{i}.pth").
Each trained model have a corresponding config file, with its hyperparams and changed in the architecture. 

A provided cell for using the models is provided in the notebook file. 

## Folders
* config: config files with the hyperparams used in each model. 
* saved_models: all checkpoint of the trained models. 

## Reference

[1] Document of the spacy model used for preprocces the data - https://spacy.io/models/en  
[2] Previous work on the dataset -  https://www.kaggle.com/code/bryamblasrimac/suicidal-prediction-multiplemodels-accuracy-73/notebook#5.-Preprocessing 
[3] Dataset - https://www.kaggle.com/datasets/subhranilpaul/suicidal-mental-health-review


## License ?

This project is open source and available under the [License](LICENSE).

