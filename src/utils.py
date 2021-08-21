import numpy as np
from sentence_transformers import  util, SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def get_sentence_embeddings(model, corpus : list ) :
    '''
    input = model name in list (multiple model)
            corpus in list 
    output = list of embeddings
    '''

    return model.encode(corpus)


def get_similarity_score(corpus_vector, sentence_vector):

    '''
    input = list of sentence ebeddings
    output = similarity score
    '''

    return(util.pytorch_cos_sim(corpus_vector, sentence_vector))


def test_model(models, dataframe, corpus, sentence):

    '''
    input = model to test, test dataset, corpus, sentence 
    output = similarity score in list
    '''

    similarity_score_dict, similarity_score= { },[]
    for model in models:
        model_name = model
        model = SentenceTransformer(model)

        corpus_embedding = get_sentence_embeddings(model, list(dataframe[corpus]))

        sentence_embedding = get_sentence_embeddings(model, list(dataframe[sentence]))

        for  index in range(len(corpus_embedding)):
        
            similarity_score.append(np.array(util.pytorch_cos_sim(corpus_embedding[index], sentence_embedding[index])))
                                         
            similarity_score_dict[model_name] = np.array(similarity_score)
    
    return similarity_score_dict
