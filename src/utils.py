import numpy as np
from sentence_transformers import  util, SentenceTransformer



def get_sentence_embeddings(model, sentence : list ) :
    
    '''
    model = trained machine learning/ deep learning model
    -----------------------------------------------------
    convert sentence into embeddings
    -----------------------------------------------------
    return embeddings of give sentence
    '''

    return model.encode(sentence)


def get_similarity_score(sentence_vector_1, sentence_vector_2):

    '''
    input = list of sentence ebeddings
    output = similarity score
    '''

    return(util.pytorch_cos_sim(sentence_vector_1, sentence_vector_2))


def test_model(models, dataframe, sentance_1, sentence_2):

    '''
    input = model to test, test dataset, corpus, sentence 
    output = similarity score in list
    '''

    similarity_score_dict, similarity_score= { },[]
    for model in models:
        model_name = model
        model = SentenceTransformer(model)
        corpus_embedding = get_sentence_embeddings(model, list(dataframe[sentance_1]))
        sentence_embedding = get_sentence_embeddings(model, list(dataframe[sentence_2]))
        for  index in range(len(corpus_embedding)):
            similarity_score.append(np.array(util.pytorch_cos_sim(corpus_embedding[index], sentence_embedding[index])))                           
            similarity_score_dict[model_name] = np.array(similarity_score).reshape(1,)
    
    return similarity_score_dict
