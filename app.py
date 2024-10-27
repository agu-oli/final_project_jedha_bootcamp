import os
import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
import fasttext

import re
import string
from unidecode import unidecode

import pandas as pd
import numpy as np
import pickle
import joblib
import io
import warnings
warnings.filterwarnings('ignore')

print('Jedha2')


print('Jedha3')


from transformers import BertTokenizerFast
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

print('Jedha4')

# Importation du model de finBERT
finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)

bert_tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

nlp_finbert = pipeline("sentiment-analysis", model=finbert, tokenizer=bert_tokenizer)


print('Jedha5')
  

print('Jedha5bis') 

def tokenize(text):
    #Remove patterns like @BillyM2k which are twitter accounts
    wp = re.compile(r"\w+") 

    text = re.sub(r'@\w+', '', text)  #Normalize text: remove accents, and tokenize


    tokens = wp.findall(unidecode(text).lower()) #Remove patterns and lower case

    tokens = [t for t in tokens if not t.isdigit()]  # Remove digits

    return tokens

model_path = 'cc.en.300.bin'

fasttext_model = fasttext.load_model(model_path)

def get_embeddings(text):
  return fasttext_model.get_sentence_vector(text)



# Load the Random Forest model from the file
rf_model_loaded = joblib.load('rf_model.joblib')


print('Jedha6')

def preprocessing(text):
    df = pd.DataFrame(data={'text': [text],
                            })
    X = df['text']
    X = list(X)

    results = nlp_finbert(X)

    label_values = [d['label'] for d in results]
    df['sentiment'] = label_values

    # Création d'une colonne avec en numérique le type de sentiment

    sentiment_numerique = {
        'Neutral': 0,
        'Negative': 1,  # Adjusting this
        'Positive': 2   # Adjusting this
    }

    print('Jedha7')
    # Ajout de la nouvelle colonne "valeur_sentiment"
    df['sentiment_label'] = df['sentiment'].map(sentiment_numerique)

    dataset = df.copy()

    dataset["text"] = dataset["text"].astype(str).fillna("")
    dataset["text"] = dataset["text"].tolist()


    dataset['text'] = [" ".join(tokenize(doc)) for doc in dataset['text']]


    print('Jedha10')

    #Embedding with fasttext

    dataset['embeddings'] = dataset['text'].apply(get_embeddings)
    print(dataset)


    X = dataset[['sentiment_label', 'embeddings']]

    # Flatten embeddings into a DataFrame
    embeddings_df = pd.DataFrame(np.vstack(X['embeddings'].to_numpy()))
    X = pd.concat([X.drop(columns=['embeddings']), embeddings_df], axis=1)

    X.columns = X.columns.astype(str)

    #Prediction

    # Use the loaded model for predictions
    prediction = rf_model_loaded.predict(X)

    result = prediction[0]

    #print('Jedha11')
    print(prediction)
    print(result)
    return result





description = """
## JEDHA's Data Science bootcamp project

## Predict Tesla's stocks variation from Elon Musk Tweets

#### In this API you can predict if a Elon Musk tweet will have a neutral, positive or negative impact on Tesla actions.

* Endpoints: 
            /predict_impact: insert the raw text of an Elon Musk's and obtain a prediction of the Tesla stock evolution
            
* HTTP Method: POST

"""

app = FastAPI(
    title="👨‍💼 Elon Musk Tweets Impact Prediction on Tesla's action API",
    description=description,
    version="0.1",
    #openapi_tags=tags_metadata
)

@app.post("/predict_impact", tags=["predict_impact"])
async def PredictNewTweet(text: str
                          ):
    """
    Process a new text from Elon Musk and return a prediction of the evolution of the Tesla stock.
    """
    result = preprocessing(text)
    result = int(result)  # Convert numpy.int64 to Python int


    if result == 2:  # Assuming 2 for Positive
        prediction = "positive"
    elif result == 1:  # Assuming 1 for Negative
        prediction = "neutral"
    elif result == 0:  # Assuming 0 for Neutral
        prediction = "negative"
    else:
        prediction = "Unknown"  # This should theoretically not be reached

    message = {"prediction": prediction,
               "message": "The estimated evolution of the Tesla stock is: " + prediction}

    return message

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4000)



