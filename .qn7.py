import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score,f1_score

df = pd.read_csv("./20newsgroups.csv")
df = df.dropna()
X = df["text"]
y = df["target"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

multinomial = make_pipeline(
    CountVectorizer(),
    MultinomialNB()
)

multinomial.fit(X_train,y_train)
y_multi_predict = multinomial.predict(X_test)

bernoulli = make_pipeline(
    CountVectorizer(binary=True),
    BernoulliNB()
)

bernoulli.fit(X_train,y_train)
y_berno_predict = bernoulli.predict(X_test)

print("MultinomialNB :")
print(f"Accuracy : {accuracy_score(y_test,y_multi_predict)}")
print(f"F1-Score : {f1_score(y_test,y_multi_predict,average="weighted")}")

print("BernoulliNB :")
print(f"Accuracy : {accuracy_score(y_test,y_berno_predict)}")
print(f"F1-Score : {f1_score(y_test,y_berno_predict,average="weighted")}")
