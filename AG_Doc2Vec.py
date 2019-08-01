import pandas as pd
import numpy as np
import spacy
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from spacy.lang.en import English
import sys

class IntentVectorization():
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.max_epochs = 50
        self.vec_size = 20
        self.alpha = 0.025
        pass

    def preprocessor(self):
        if len(sys.argv) < 2:
            print("Expects <python> <program.py> <filename.csv>")
            exit()
        df = pd.read_csv(sys.argv[1], header = None)
        df = df.iloc[:,2]
        data = [string for string in df]
        # creating tagged data, also performs tokenization and lemmatization
        tagged_data = [TaggedDocument(words=[token.lemma_ for token in self.nlp(_d.lower()) 
                        if not token.is_stop], tags=[str(i)]) 
                        for i, _d in enumerate(data)]
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
                        workers = 4,
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
    # building model
    model = intent_vectorization.build_model(tagged_data)
    # path to save/load model
    model_path = "Models\intentModel"
    # checking if the model got saved
    if intent_vectorization.save_model(model, model_path):
        print("Model Saved")
    else:
        print("Model Not saved")

    # loading saved model
    model= Doc2Vec.load(model_path + ".model")

    # testing 
    #test_data = word_tokenize("I would like to know about microsoft windows XP, microsoft released their product xp this october, what are your thoughts on this? Maybe the updates will be slow".lower())
    #test_data = word_tokenize("Microsoft xp update".lower())
    test_data = word_tokenize("Space exploration programs have really found new possibilities and gave new insights into the space. It has broadened our horizon and also makes sure that we know where we come from. I would love to know more about space and existence.".lower())
    v1 = model.infer_vector(test_data)
    print("V1_infer", v1)

    # to find most similar doc using tags (returns 10 most similar docs according to cosine similarity)
    similar_doc = model.docvecs.most_similar(positive=[v1])
    print(similar_doc)
