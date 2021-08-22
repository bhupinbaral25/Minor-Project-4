import re
import nltk
import scipy
import string
import numpy as np
from nltk.corpus import stopwords
from scipy.stats import pearsonr
from scipy.spatial import distance
from nltk.stem import WordNetLemmatizer


class Utils:
    """
    A class containing utilities methods.

    ...

    Methods
    -------
    remove_punctuation(text)
        Removes punctuation(!”#$%&'()*+,-./:;?@[\]^_`{|}~) from a text and returns it.

    tokenization(text):
        Splits and returns the word list from a text.

    remove_stopwords(wordlist)
        Removes stopword(nltk.corpus.stopwords) from wordlist and returns.

    lemmatizer(wordlist)     
        Lemmatizes word in the wordlist and returns it.

    listtostr(wordlist)
        Creates sentences from wordlist.

    preprocess(dataframe, column)
        Preprocess the column of dataframe. 
    
    similarity_score(sentence1_vectorized, sentence2_vectorized):
        Calculates and returns cosine similarity scores.

    pearson_coeff_score(cls, sim_scores, other_sim_scores)
        Calculates and returns pearson coefficient score and returns it.

    cosine_distance(cls, vector1, vector2)
        Calculates and returns cosine distance.

    split_sentence(cls, sentence)
        Splits the sentence to word list and returns it.

    """
    @classmethod
    def remove_punctuation(cls, text):
        """
        Removes punctuation(!”#$%&'()*+,-./:;?@[\]^_`{|}~) from a text and returns it.

        Parameters
        ----------
        text(str) : text for stopword removal

        Returns
        -------
        processed_text(str) : sentence with punctuation removed
        """

        processed_text = re.sub(r"[!”#$%&'()*+,-./:;?@[\]^_`{|}~]+", " ", text)
        return processed_text

    @classmethod
    def tokenization(cls, text):
        """
        Splits and returns the word list from a text.

        Parameters
        ----------
        text(str) : text split

        Returns
        -------
        tokens(str) : word list created from text
        """

        tokens = text.strip().split(" ")
        return tokens

    @classmethod
    def remove_stopwords(cls, wordlist):
        """
        Removes stopword(nltk.corpus.stopwords) from wordlist and returns it.

        Parameters
        ----------
        wordlist(list) : list of words

        Returns
        -------
        processed_list(list) : word list with stopwords removed
        """  

        stopwords_list = stopwords.words('english')
        processed_list = [word for word in wordlist if word not in stopwords_list]
        return processed_list

    @classmethod
    def lemmatizer(cls, wordlist):
        """
        Lemmatizes word in the wordlist and returns it.

        Parameters
        ----------
        wordlist(list) : list of words

        Returns
        -------
        lemmatized_list(list) : word list with words lemmatized
        """  

        wordnet_lemmatizer = WordNetLemmatizer()
        lemmatized_list = [wordnet_lemmatizer.lemmatize(
            word) for word in wordlist]
        return lemmatized_list

    @classmethod
    def listtostr(cls, wordlist):
        """
        Creates sentences from wordlist.

        Parameters
        ----------
        wordlist(list) : list of words

        Returns
        -------
        string(str) : sentence created from wordlist
        """  

        string = " ".join([str(word) for word in wordlist])
        return string
        

    @classmethod
    def preprocess(cls, dataframe, column):
        """
        Preprocess the column of dataframe. 

        Parameters
        ----------
        dataframe(pandas.core.frame.DataFrame) : pandas dataframe
        column(str) : column name

        Returns
        -------
        None
        """  
        
        # Remove punctuation
        dataframe[column] = dataframe[column].apply(lambda text: Utils.remove_punctuation(text))
        # Convert lowercase
        dataframe[column] = dataframe[column].apply(lambda text: text.lower())
        # Tokenization
        dataframe[column] = dataframe[column].apply(lambda text: Utils.tokenization(text))
        # Remove stopwords
        dataframe[column] = dataframe[column].apply(lambda wordlist: Utils.remove_stopwords(wordlist))
        # Lemmatization
        dataframe[column] = dataframe[column].apply(lambda wordlist: Utils.lemmatizer(wordlist))
        # List of words to sentences
        dataframe[column] = dataframe[column].apply(lambda wordlist: Utils.listtostr(wordlist))

    @classmethod
    def similarity_score(cls, sentence1_vectorized, sentence2_vectorized):
        """
        Calculates and returns cosine similarity scores.

        Parameters
        ----------
        sentence1_vectorized(scipy.sparse.csr.csr_matrix) : matrix containing embedding vectors
        sentence2_vectorized(scipy.sparse.csr.csr_matrix) : matrix containing embedding vectors

        Returns
        -------
        sim_scores(list): cosine similarity scores
        """  

        sim_scores = []
        length = 0
        if isinstance(sentence1_vectorized, scipy.sparse.csr_matrix):
            # We have sparse matrix
            length = sentence1_vectorized.shape[0]
            for idx in range(length):

                vec1 = sentence1_vectorized[idx].toarray()
                vec2 = sentence2_vectorized[idx].toarray()
                sim = 1 - Utils.cosine_distance(vec1, vec2)
                if np.isnan(sim):
                    sim_scores.append(0)
                else:
                    sim_scores.append(sim)
        else:
            length = len(sentence1_vectorized)
            for idx in range(length):

                vec1 = sentence1_vectorized[idx]
                vec2 = sentence2_vectorized[idx]
                sim = 1 - Utils.cosine_distance(vec1, vec2)
                if np.isnan(sim):
                    sim_scores.append(0)
                else:
                    sim_scores.append(sim)
        return sim_scores

    @classmethod
    def pearson_coeff_score(cls, sim_scores, other_sim_scores):
        """
        Calculates and returns pearson coefficient score and returns it.

        Parameters
        ----------
        sim_scores(list) : list of cosine similarity scores
        other_sim_scores(list) : other list of cosine similarity scores

        Returns
        -------
        pearson_coefficient_score(numpy.float64): pearson coefficient score
        """  

        pearson_coefficient_score = round(pearsonr(sim_scores, other_sim_scores)[0], 2)
        return pearson_coefficient_score

    @classmethod
    def cosine_distance(cls, vector1, vector2):
        """
        Calculates and returns pearson coefficient score and returns it.

        Parameters
        ----------
        vector1(numpy.ndarray) : array of embedding
        vector2(numpy.ndarray) : other array of embedding

        Returns
        -------
        cos_distance(numpy.float64): cosine distance
        """  

        cos_distance = distance.cosine(vector1, vector2)
        return cos_distance

    @classmethod
    def split_sentence(cls, sentence):
        """
        Splits the sentence to word list and returns it.

        Parameters
        ----------
        sentence(str) : input sentence

        Returns
        -------
        wordlist(list): list of words created from input sentence
        """  
        
        wordlist = sentence.split()
        return wordlist
