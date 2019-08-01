from nltk.corpus import reuters
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, accuracy_score, hamming_loss
from sklearn.metrics import f1_score, precision_score, recall_score
import warnings

warnings.filterwarnings(action = 'ignore') 

class TextClassification(object):
    def __init__(self, corpusData):
        # getting all the document ids in corpusData, such as reuters
        docs = corpusData.fileids()

        # splitting into training and test docs ids
        train_docs_ids = list(filter(lambda doc: doc.startswith("train"), docs))
        test_docs_ids = list(filter(lambda doc: doc.startswith("test"), docs))

        # getting the actual data from those ids
        self.train_docs = [corpusData.raw(doc_id) for doc_id in train_docs_ids]
        self.test_docs = [corpusData.raw(doc_id) for doc_id in test_docs_ids]

        self.docs = self.train_docs + self.test_docs

        # transforming multilabels
        mlb = MultiLabelBinarizer()
        self.train_labels = mlb.fit_transform([reuters.categories(doc_id) 
                for doc_id in train_docs_ids])
        self.test_labels = mlb.transform([reuters.categories(doc_id) 
                for doc_id in test_docs_ids])

    def train_test_split(self):
        # returns data and their respective class labels split into training and test set.
        return (self.train_docs, self.test_docs, self.train_labels, 
                self.test_labels)

    def vectorizer(self, vectorizer):
                ### Count Vectorization ###
        # Count Vector is a matrix notation of the dataset 
        # in which every row represents a document from the corpus, 
        # every column represents a term from the corpus, 
        # and every cell represents the frequency count 
        # of a particular term in a particular document.

                ### TF-IDF Vectorization ###
        # TF-IDF score represents the relative importance
        # of a term in the document and the entire corpus. 
        # TF-IDF score is composed by two terms: 
        # the first computes the normalized Term Frequency (TF), 
        # the second term is the Inverse Document Frequency (IDF), 
        # computed as the logarithm of the number of 
        # the documents in the corpus divided by the number of 
        # documents where the specific term appears.

        train_docs = vectorizer.fit_transform(self.train_docs)
        test_docs = vectorizer.transform(self.test_docs)
        return train_docs, test_docs


    def classify(self, classifier, tfidf_vect_train_docs, train_labels, 
            tfidf_vect_test_docs, test_labels):
        model = OneVsRestClassifier(classifier)
        # fitting/training the model on training data
        model.fit(tfidf_vect_train_docs, train_labels)
        # testing the classifier on test data
        predictions = model.predict(tfidf_vect_test_docs)
        print("SVM Classifier Results:-----------")
        # printing accuracy report and hamming loss
        self.report(predictions, test_labels)


    def report(self, predictions, test_labels):
        # printing accuracy of the results
        print('accuracy %s' % accuracy_score(predictions, test_labels))
        # complete classification report
        print(classification_report(test_labels, predictions))
        # hamming loss
        print(hamming_loss(test_labels, predictions))
        print(end="\n\n")


if __name__ == "__main__":
    #reading data in proper format
    text_classification = TextClassification(reuters)
    train_docs, test_docs, train_labels, test_labels = \
        text_classification.train_test_split()
    
    # representing data as Vectors
    count_vect_train_docs, count_vect_test_docs = text_classification.vectorizer(
        CountVectorizer(analyzer='word', stop_words='english')) # Vectorizing with CountVectorizer
    tfidf_vect_train_docs, tfidf_vect_test_docs = text_classification.vectorizer(
        TfidfVectorizer(analyzer='word', stop_words='english')) # Vectorizing with TfidfVectorizer

    # SVM classifier results 
    print("When Used with TFIDF Vectorizer")
    text_classification.classify(LinearSVC(random_state=42), tfidf_vect_train_docs, 
            train_labels, tfidf_vect_test_docs, test_labels)     # Used TFIDF Vectorizer
    print("When Used with Count Vectorizer")
    text_classification.classify(LinearSVC(random_state=42), count_vect_train_docs, 
            train_labels, count_vect_test_docs, test_labels) # Used Count Vectorizer

    # Multinomial Naive Bayes Classifier Results
    print("When Used with TFIDF Vectorizer")
    text_classification.classify(MultinomialNB(), tfidf_vect_train_docs, 
            train_labels, tfidf_vect_test_docs, test_labels) # Used TFIDF Vectorizer
    print("When Used with Count Vectorizer")
    text_classification.classify(MultinomialNB(), count_vect_train_docs, 
            train_labels, count_vect_test_docs, test_labels) # Used Count Vectorizer

    # Logistic Regression Model Results
    print("When Used with TFIDF Vectorizer")
    text_classification.classify(LogisticRegression(), tfidf_vect_train_docs, 
            train_labels, tfidf_vect_test_docs, test_labels) # Used TFIDF Vectorizer
    print("When Used with Count Vectorizer")
    text_classification.classify(LogisticRegression(), count_vect_train_docs, 
            train_labels, count_vect_test_docs, test_labels) # Used Count Vectorizer

