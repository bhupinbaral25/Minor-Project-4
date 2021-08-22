import numpy as np
from helpers import Utils
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


class CountVectorizerEmbedding:
    """
    A class used to create sentence embedding using count vectorizer.

    ...

    Attributes
    ----------
    vectorizer : CountVectorizer
        A countvectorizer object


    Methods
    -------
    preprocess(df, column_name) 
        Preprocess the column of dataframe. 

    fit(df, column_name)
        Creates count vectorizer and fits the dataframe column.

    get_sentence_embedding(df, column_name):
        Creates and returns the sentence embeddings for the dataframe column.

    similarity_score(embeddings, other_embeddings):
        Computes and returns the cosine similarity score.

    pearson_correlation(self, sim_scores, other_sim_scores)
        Computes and returns the pearson correlation coefficient.

    """

    def preprocess(self, df, column_name):
        """
        Preprocess the column of dataframe.

        Parameters
        ----------
        df(pandas.core.frame.DataFrame) : pandas dataframe
        column_name(str) : column name

        Returns
        -------
        None
        """

        Utils.preprocess(df, column_name)


    def fit(self, df, column_name):
        """
        Creates count vectorizer and fits the dataframe column.

        Parameters
        ----------
        df(pandas.core.frame.DataFrame) : pandas dataframe
        column_name(str) : column name

        Returns
        -------
        None
        """

        self.vectorizer = CountVectorizer()
        self.vectorizer.fit(df[column_name])
    

    def get_sentence_embedding(self, df, column_name):
        """
        Creates and returns the sentence embeddings for the dataframe column.

        Parameters
        ----------
        df(pandas.core.frame.DataFrame) : pandas dataframe
        column_name(str) : column name

        Returns
        -------
        embeddings(scipy.sparse.csr.csr_matrix) : matrix of sentence embeddings
        """

        embeddings = self.vectorizer.transform(df[column_name])
        return embeddings


    def similarity_score(self, embeddings, other_embeddings):
        """
        Computes and returns the cosine similarity score.

        Parameters
        ----------
        embeddings(scipy.sparse.csr.csr_matrix) :  matrix of sentence embeddings
        other_embeddings(scipy.sparse.csr.csr_matrix) :  matrix of sentence embeddings

        Returns
        -------
        score(list) : list of pairwise similarity scores 
        """   

        score = Utils.similarity_score(embeddings, other_embeddings)
        return score


    def pearson_correlation(self, sim_scores, other_sim_scores):
        """
        Computes and returns the pearson correlation coefficient.

        Parameters
        ----------
        sim_scores(list) : list of sentence similarity scores
        other_sim_scores(list) : list of sentence similarity scores

        Returns
        -------
        pearson_coeff(numpy.float64) : pearson correlation coefficient
        """

        pearson_coeff = Utils.pearson_coeff_score(sim_scores, other_sim_scores)
        return pearson_coeff




class TfidfEmbedding:
    """
    A class used to create sentence embedding using tfidf vectorizer.

    ...

    Attributes
    ----------
    vectorizer : TfidfVectorizer
        A tfidf vectorizer object


    Methods
    -------
    preprocess(df, column_name) 
        Preprocess the column of dataframe. 

    fit(df, column_name)
        Creates tfidf vectorizer and fits the dataframe column.

    get_sentence_embedding(df, column_name):
        Creates and returns the sentence embeddings for the dataframe column.

    similarity_score(embeddings, other_embeddings):
        Computes and returns the cosine similarity score.

    pearson_correlation(self, sim_scores, other_sim_scores)
        Computes and returns the pearson correlation coefficient.

    """

    def preprocess(self, df, column_name):
        """
        Preprocess the column of dataframe. 

        Parameters
        ----------
        df(pandas.core.frame.DataFrame) : pandas dataframe
        column_name(str) : column name

        Returns
        -------
        None
        """

        Utils.preprocess(df, column_name)

    def fit(self, df, column_name):
        """
        Creates tfidf vectorizer and fits the dataframe column.

        Parameters
        ----------
        df(pandas.core.frame.DataFrame) : pandas dataframe
        column_name(str) : column name

        Returns
        -------
        None
        """

        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(df[column_name])
    
    def get_sentence_embedding(self, df, column_name):
        """
        Creates and returns the sentence embeddings for the dataframe column.

        Parameters
        ----------
        df(pandas.core.frame.DataFrame) : pandas dataframe
        column_name(str) : column name

        Returns
        -------
        embeddings(scipy.sparse.csr.csr_matrix) : matrix of sentence embeddings
        """

        embeddings = self.vectorizer.transform(df[column_name])
        return embeddings
        
    def similarity_score(self, embeddings, other_embeddings):
        """
        Computes and returns the cosine similarity score.

        Parameters
        ----------
        embeddings(scipy.sparse.csr.csr_matrix) :  matrix of sentence embeddings
        other_embeddings(scipy.sparse.csr.csr_matrix) :  matrix of sentence embeddings

        Returns
        -------
        score(list) : list of pairwise similarity scores 
        """   

        score = Utils.similarity_score(embeddings, other_embeddings)
        return score
        
    def pearson_correlation(self, sim_scores, other_sim_scores):
        """
        Computes and returns the pearson correlation coefficient.

        Parameters
        ----------
        sim_scores(list) : list of sentence similarity scores
        other_sim_scores(list) : list of sentence similarity scores

        Returns
        -------
        pearson_coeff(numpy.float64) : pearson correlation coefficient
        """

        pearson_coeff = Utils.pearson_coeff_score(sim_scores, other_sim_scores)
        return pearson_coeff



class GloveEmbedding:
    """
    A class used to create sentence embedding using glove word embeddings.

    ...

    Attributes
    ----------
    file_path : str
        A path to saved glove embedding model.
    model : dict
        Word and glove embedding pair in a dictionary.
    dim_embedding : int
        Dimension of word embedding in glove model


    Methods
    -------
    load_glove() 
        Loads pretrained word glove embedding.

    glove_sentence_embedding(df, column_name)
        Creates and returns sentence embedding using word glove embedding for a dataframe column.

    compute_embedding_using_avg(word_list)
        Computes and returns the sentence embeddings from word embedding using averaging technique.

    preprocess(df, column_name)
        Preprocess the column of dataframe. 

    similarity_score(embeddings, other_embeddings):
        Computes and returns the cosine similarity score.

    pearson_correlation(self, sim_scores, other_sim_scores)
        Computes and returns the pearson correlation coefficient.

    """

    def __init__(self, file_path):
        """
        Initializes and creates GloveEmbedding class object and loads glove model.

        Parameters
        ----------
        file_path(str) : A path to saved glove embedding model.
        """

        self.file_path = file_path
        self.load_glove()

    def load_glove(self):
        """
        Loads pretrained word glove embedding.

        Returns
        -------
        None
        """

        with open(self.file_path, encoding="utf8" ) as f:
            content = f.readlines()
        self.model = {}
        # Loop through the content of a file
        for line in content:
            current_line = line.split()
            word = current_line[0]
            embedding = np.array([float(val) for val in current_line[1:]])
            self.model[word] = embedding
        for word, embedding  in self.model.items():
            self.dim_embedding = len(self.model[word])
            break

    def glove_sentence_embedding(self, df, column_name):
        """
        Creates and returns sentence embedding using word glove embedding for a dataframe column.

        Parameters
        ----------
        df(pandas.core.frame.DataFrame) : pandas dataframe
        column_name(str) : column name

        Returns
        -------
        embedding_matrix(list) : list of sentence embeddings
        """
        
        tokenized_series = df[column_name].apply(lambda text: Utils.split_sentence(text))
        embedding_matrix = []
        for list in tokenized_series.iteritems():
            word_list = list[1]
            embedded_sentences = self.compute_embedding_using_avg(word_list)
            embedding_matrix.append(embedded_sentences)
        return embedding_matrix
        
    
    def compute_embedding_using_avg(self, word_list):
        """
        Computes and returns the sentence embeddings from word embedding using averaging technique.

        Parameters
        ----------
        word_list(list) : list of words
        
        Returns
        -------
        embedding_matrix(numpy.ndarray) : sentence embeddings, calculated using average of word embedding.
        """

        word_embeddings = [self.model[word] for word in word_list if word in self.model.keys()]
        if not len(word_embeddings):
            # If we have no word embeddings for a whole sentence, create embeddings of zeros.
            word_embeddings = np.zeros(self.dim_embedding)
        # Caculate sentence embedding using average of each word embeddings of a sentence
        sentence_embedding = np.mean(word_embeddings, axis = 0)
        return sentence_embedding

    def preprocess(self, df, column_name):
        """
        Preprocess the column of dataframe. 

        Parameters
        ----------
        df(pandas.core.frame.DataFrame) : pandas dataframe
        column_name(str) : column name

        Returns
        -------
        None
        """

        Utils.preprocess(df, column_name)
        

    def similarity_score(self, embeddings, other_embeddings):
        """
        Computes and returns the cosine similarity score.

        Parameters
        ----------
        embeddings(scipy.sparse.csr.csr_matrix) :  matrix of sentence embeddings
        other_embeddings(scipy.sparse.csr.csr_matrix) :  matrix of sentence embeddings

        Returns
        -------
        score(list) : list of pairwise similarity scores 
        """   

        score = Utils.similarity_score(embeddings, other_embeddings)
        return score

    def pearson_correlation(self, sim_scores, other_sim_scores):
        """
        Computes and returns the pearson correlation coefficient.

        Parameters
        ----------
        sim_scores(list) : list of sentence similarity scores
        other_sim_scores(list) : list of sentence similarity scores

        Returns
        -------
        pearson_coeff(numpy.float64) : pearson correlation coefficient
        """

        pearson_coeff = Utils.pearson_coeff_score(sim_scores, other_sim_scores)
        return pearson_coeff