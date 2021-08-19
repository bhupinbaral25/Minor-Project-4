import pandas as pd
import numpy as np 
from sentence_transformers import SentenceTransformer
from utils import get_sentence_embeddings

corpus = ['My name is Bhupin Baral','I live in Kathmandu Nepal']

models = ['paraphrase-MiniLM-L6-v2', 'nli-distilroberta-base-v2']
for model in models:

    model = SentenceTransformer(model)
    sentence_embeddings = get_sentence_embeddings(model, corpus)
print(sentence_embeddings)


df = pd.read_csv('../data/sts_test.csv')

print(df)