# Suicidal Prediction Transformer

This repository contains code for a Suicidal Prediction Transformer, which is a Transformer-based model trained from scratch, to predict suicidal tendencies based on textual data.


## Preprocessing

Before using the model, it is important to preprocess the data. The preprocessing steps include:
- Tokenization: Splitting the text into individual words or subwords.
- Padding: Ensuring that all input sequences have the same length by adding padding tokens.
- Encoding: Converting the text into numerical representations that can be understood by the model.


## Data Preprocessing

The data preprocessing steps include:
1. removing non-alphabetic characters, converting to lowercase, removing stopwords and punctuation, and lemmatizing the remaining tokens

2. **Tokenization**: The text is split into individual words or tokens. This is a crucial step as the model learns from these tokens.

3. **Padding**: All text sequences are padded to have the same length. This is necessary because the input to the model needs to be of the same size.

4. **Encoding**: The tokens are converted into numerical values or indices. This is done because the model cannot process raw text and works with numerical data.



## Model

The model used in this project is a Transformer-based text classifier. It consists of an embedding layer, a positional encoding layer, a Transformer encoder, and a linear layer for classification.

The embedding layer converts the input indices into dense vectors of fixed size. The positional encoding layer adds positional information to the input embeddings. The Transformer encoder processes the input data and the output is passed to the linear layer for classification.

The model is trained using the Adam optimizer and the Binary Cross Entropy with Logits loss function.

## Usage

To use this project, you need to have Python and PyTorch installed. You can then run the `deep_learning_project.ipynb` notebook to train and evaluate the model.

After preprocces once, the code saves the preproccesed data for later use. 
Just skip the preprocces section and load the saved preproccesed data. 

The repository provides a preproccesed data file. 


## Evaluate

10 pretrained models are provided, ("model_checkpoint_{i}.pth").



## License

This project is open source and available under the [MIT License](LICENSE).





## Classes

- `SuicidalPredictionTransformer`: The main class that implements the Suicidal Prediction Transformer model. It takes in text data as input and outputs predictions for suicidal tendencies.
