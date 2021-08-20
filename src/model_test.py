import pandas as pd


from utils import test_model

df = pd.read_csv('../data/sts_test.csv')


models = ['paraphrase-MiniLM-L6-v2', 
        'nli-distilroberta-base-v2']
      

similarity_score = test_model(models, df, df['sent_1'].loc[:1], df['sent_2'].loc[:1])

print(similarity_score)
