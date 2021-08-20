import numpy as np
from sentence_transformers import  util, SentenceTransformer
'''
input = model name in list (multiple model)
        corpus in list 
output = list of embeddings
'''

def get_sentence_embeddings(model, corpus : list ):
    
    return model.encode(corpus)
'''
input = list of sentence ebeddings
output = similarity score
'''

def get_similarity_score(corpus_vector, sentence_vector):

    return(util.pytorch_cos_sim(corpus_vector, sentence_vector))

'''
input = model to test, test dataset, corpus, sentence 
output = similarity score in list
'''

def test_model(models, test_data, corpus, sentence):
    similarity_score_dict = { }
    for model in models:
        model_name = model
        model = SentenceTransformer(model)

        corpus_embedding = get_sentence_embeddings(model, list(test_data[corpus]))

        sentence_embedding = get_sentence_embeddings(model, list(test_data[sentence]))

        for  index in range(len(test_data[corpus])):
        
            similarity_score = np.array(util.pytorch_cos_sim(corpus_embedding[index], sentence_embedding[index]))                                   
            similarity_score_dict[model_name] = similarity_score
    
    return similarity_score
