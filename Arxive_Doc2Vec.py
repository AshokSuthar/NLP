import pandas as pd
import numpy as np
import spacy
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from spacy.lang.en import English
import sys

class IntentVectorization():
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'tagger'])
        self.max_epochs = 50
        self.vec_size = 20
        self.alpha = 0.025

    def preprocessor(self):
        if len(sys.argv) < 2:
            print("Expects <python> <program.py> <filename.csv>")
            exit()
        data_df = pd.read_csv(sys.argv[1])
        print(data_df.head())
        data_df["summary"] = data_df["summary"].map(lambda x : x.replace("\n", " "))
        # creating tagged data, also performs tokenization and lemmatization
        tagged_data = [TaggedDocument(words=[token.lemma_ for token in self.nlp(_d.lower()) 
                        if not token.is_stop and len(token.text) > 2], tags=[str(i)]) 
                        for i, _d in enumerate(data_df["summary"])]
        return tagged_data

    def save_tagged_csv(self, tagged_data, outfile):
        try:
            df = pd.DataFrame(tagged_data)
            df.to_csv(outfile + ".csv", header=None, index=None)
            return True
        except:
            return False

    def build_model(self, tagged_data):
        # initializing model parameters
        model = Doc2Vec(vector_size = self.vec_size,
                        alpha = self.alpha, 
                        min_alpha = 0.00025,
                        min_count = 2,
                        workers = 8,
                        dm = 1)

        # building vocab
        model.build_vocab(tagged_data)

        # training model
        for epoch in range(self.max_epochs):
            print('iteration {0}'.format(epoch))
            model.train(tagged_data,
                        total_examples=model.corpus_count,
                        epochs=model.epochs)
            # decrease the learning rate
            model.alpha -= 0.0002
            # fix the learning rate, no decay
            model.min_alpha = model.alpha
        return model
    
    def save_model(self, model, outfile):
        try:
            model.save(outfile + ".model")
            return True
        except:
            return False

if __name__ == "__main__":
    # class object initialization
    intent_vectorization = IntentVectorization()
    # calling preprocessor() to create and return tagged_data to train the model
    tagged_data = intent_vectorization.preprocessor()
    # save tagged_csv 
    if intent_vectorization.save_tagged_csv(tagged_data, "arxiveTagged"):
        print("taggedFile Saved.")
    else:
        print("taggedFile Not Saved.")
    # building model
    model = intent_vectorization.build_model(tagged_data)
    # path to save/load model
    model_path = "Models\intentModelArxive"
    # checking if the model got saved
    if intent_vectorization.save_model(model, model_path):
        print("Model Saved")
    else:
        print("Model Not saved")

    # loading saved model
    model= Doc2Vec.load(model_path + ".model")

    # testing 
    test_data = "Deep Neural networks for automation has alot of benefits. Neural networks can find answers to the problems which are way complex to functionally program"
    test_tokenized = [token.lemma_ for token in intent_vectorization.nlp(test_data.lower()) 
                        if not token.is_stop and len(token.text) > 2]
    print(test_tokenized)
    v1 = model.infer_vector(test_data)
    print("V1_infer", v1)

    # to find most similar doc using tags (returns 10 most similar docs according to cosine similarity)
    similar_doc = model.docvecs.most_similar(positive=[v1])
    print(similar_doc)
