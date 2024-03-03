import nltk
import random
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
documents=[(list(movie_reviews.words(fileid)),category)
           for category in movie_reviews.categories()
           for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)
stop_words=set(stopwords.words('english'))
lemmatizer=WordNetLemmatizer()
print(stop_words)
def preprocess(doc):
    words=[lemmatizer.lemmatize((word.lower)) for word in doc if word.isalpha() and word.lower() not in stop_words]
    return ' '.join(words)
preprocessed_documents=[(preprocess(words), category) for words, category in documents]
x=[doc[0] for doc in preprocessed_documents]
y=[category for _, category in preprocessed_documents]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
tfidt_vectorizer=TfidfVectorizer(max_features=3000)
x_train_tfidt=tfidt_vectorizer.fit_transform((x_train))
x_test_tfidt=tfidt_vectorizer.transform(x_test)
classifier=LinearSVC()
classifier.fit(x_train_tfidt,y_train)
y_pred=classifier.predict(x_test_tfidt)
accuracy=accuracy_score(y_test,y_pred)
print('accuracy: ',accuracy)