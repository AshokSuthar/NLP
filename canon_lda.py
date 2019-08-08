import pandas as pd
import spacy
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import re
import sys

class TextPreprocessing():
    def __init__(self, filename):
        # initializing spacy nlp module
        self.nlp = spacy.load('en_core_web_sm', 
            disable=['ner', 'parser', 'tagger'])
        # loading text from file
        with open(filename, mode = 'r', encoding = "utf8") as f:
            self.text = f.read()  

    def clean_text(self, sep, min_len):
        # split into words by white space
        paras = self.text.split(sep)
        df = pd.DataFrame(paras, columns = ["content"])
        data_df =  df.copy()
        data_df["content"] = data_df["content"].map(
            lambda x : re.sub(r'[^a-zA-Z,]+', " ", x))
        data_df =  data_df[(data_df.content.str.len() > min_len)]
        return data_df
    
    def preprocess(self, text):
        result = [token.lemma_ for token in self.nlp(text.lower()) 
                if not token.is_stop and len(token.text) > 2]
        return result

if __name__ == "__main__":
    text_preprocessing = TextPreprocessing(sys.argv[1])
    df = text_preprocessing.clean_text("\n\n", 100)
    #print(df)
    processed_df = df["content"].map(text_preprocessing.preprocess)
    print(processed_df)
    dictionary = Dictionary(processed_df)
    print(len(dictionary))
    # filter dictionary if required
    '''
    dictionary.filter_extremes(no_below=3, no_above=0.5)
    #print(len(dictionary))
    '''
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_df]
    # Preview Bag Of Words for our sample preprocessed document.
    '''
    bow_doc_900 = bow_corpus[900]
    for i in range(len(bow_doc_900)):
        print("Word {} (\"{}\") appears {} time.".format(bow_doc_900[i][0], 
            dictionary[bow_doc_900[i][0]], bow_doc_900[i][1]))
    '''
    lda_model = LdaModel(bow_corpus, num_topics=10, id2word= dictionary)
    for idx, topic in lda_model.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))

    # testing. Getting topic probabilities for test docs.
    other_texts = [
        ['auto', 'power', 'battery', 'information'],
        ['survey', 'response', 'eps'],
        ['human', 'system', 'computer']
    ]
    other_corpus = [dictionary.doc2bow(text) for text in other_texts]
    unseen_doc = other_corpus[0]
    vector = lda_model[unseen_doc]  # get topic probability distribution for a document
    print(vector)