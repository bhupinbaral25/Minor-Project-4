import pandas as pd
import numpy as np 
from sentence_transformers import SentenceTransformer
from utils import get_sentence_embeddings, get_cosine_similarity

df = pd.read_csv('../data/sts_test.csv')


models = ['paraphrase-MiniLM-L6-v2', 
        'nli-distilroberta-base-v2']
similarity_score_dict = { }
for model in models:
    model = SentenceTransformer(model)
    for  index in range(len(df)):
        similarity_score = get_cosine_similarity(get_sentence_embeddings(model, list(df['sent_2']))[index], 
                                                get_sentence_embeddings(model, list(df['sent_1']))[index])
    
    similarity_score_dict[model] = similarity_score