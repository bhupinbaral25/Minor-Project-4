import pandas as pd

from src import utils

if __name__ == '__main__':

    df = pd.read_csv('../sts_dataset/sts_test.csv')

    models = ['paraphrase-MiniLM-L6-v2', 
        'nli-distilroberta-base-v2']
        
    similarity_score = utils.test_model(models, df, 'sent_1', 'sent_2')