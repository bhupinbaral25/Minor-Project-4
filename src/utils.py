from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

from sklearn.metrics.pairwise import cosine_similarity

def get_sentence_embeddings(model, corpus : list ):
    
    return model.encode(corpus)

def get_cosine_similarity(corpous_vector : list, sentence_vector : list):

    return(cosine_similarity(
    [corpous_vector],
    [sentence_vector])
)

