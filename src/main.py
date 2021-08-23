from models import CountVectorizerEmbedding,TfidfEmbedding,GloveEmbedding
import pandas as pd
from src import utils

models = ['CountVectorizerEmbedding','TfidfEmbedding','GloveEmbedding']
pretrained_model = ['paraphrase-MiniLM-L6-v2','nli-distilroberta-base-v2']

pearson_coeff_dict = { }

for model in models:
    # Reading training and testing sentences
    sts_train = pd.read_csv("../sts_dataset/sts_train.csv", usecols =["sent_1"])
    sts_test = pd.read_csv("../sts_dataset/sts_test.csv")

    if model == "CountVectorizerEmbedding":
        cv_embed = CountVectorizerEmbedding()
        cv_embed.preprocess(sts_train, "sent_1")
        cv_embed.fit(sts_train, "sent_1")
        # Preprocess test sentences
        cv_embed.preprocess(sts_test, "sent_1")
        cv_embed.preprocess(sts_test, "sent_2")
        # Get embedding with count vectorizer for test sentences
        sentences1_embedding = cv_embed.get_sentence_embedding(sts_test,"sent_1")
        sentences2_embedding = cv_embed.get_sentence_embedding(sts_test,"sent_2")
        # Calculating similarity between sentences
        sim_scores = cv_embed.similarity_score(sentences1_embedding, sentences2_embedding)
        pearson_coeff = cv_embed.pearson_correlation(sim_scores, sts_test["sim"])
        pearson_coeff_dict[model] = pearson_coeff
    elif model == "TfidfEmbedding":
        cv_embed = TfidfEmbedding()
        cv_embed.preprocess(sts_train, "sent_1")
        cv_embed.fit(sts_train, "sent_1")
        # Preprocess test sentences
        cv_embed.preprocess(sts_test, "sent_1")
        cv_embed.preprocess(sts_test, "sent_2")
        # Get embedding with tfidf for test sentences
        sentences1_embedding = cv_embed.get_sentence_embedding(sts_test,"sent_1")
        sentences2_embedding = cv_embed.get_sentence_embedding(sts_test,"sent_2")
        # Calculating similarity between sentences
        sim_scores = cv_embed.similarity_score(sentences1_embedding, sentences2_embedding)
        pearson_coeff = cv_embed.pearson_correlation(sim_scores, sts_test["sim"])
        pearson_coeff_dict[model] = pearson_coeff
    elif model == "GloveEmbedding":
        path = "../pretrained-glove-embedding/glove.6B.50d.txt"
        glv_embed = GloveEmbedding(path)
        # Preprocess test sentences
        glv_embed.preprocess(sts_test, "sent_1")
        glv_embed.preprocess(sts_test, "sent_2")
        # Get embedding with glove vectors for test sentences
        sentences1_embedding_glove = glv_embed.glove_sentence_embedding(sts_test, "sent_1")
        sentences2_embedding_glove = glv_embed.glove_sentence_embedding(sts_test, "sent_2")
        # Calculating similarity between sentences
        sim_scores = glv_embed.similarity_score(sentences1_embedding_glove, sentences2_embedding_glove)
        pearson_coeff = glv_embed.pearson_correlation(sim_scores, sts_test["sim"])
        pearson_coeff_dict[model] = pearson_coeff
        
sts_test = pd.read_csv("../sts_dataset/sts_test.csv")   
similarity_score = utils.test_model(pretrained_model, sts_test, 'sent_1', 'sent_2')
similariy_score_df = pd.DataFrame.from_dict(similarity_score)
print('Similarity Score', similarity_score_df)    
print("pearson_coeff_dict : ",pearson_coeff_dict)