import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score,log_loss

df = pd.read_csv("./20newsgroups.csv")
df = df.dropna()

X = df["text"]
y = df["target"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

mle_model = make_pipeline(
    CountVectorizer(),
    MultinomialNB(alpha=1e-10)
)

mle_model.fit(X_train,y_train)
y_mle_predict = mle_model.predict(X_test)

print("MLE Metrics : \n")
print(f"Accuracy : {accuracy_score(y_test,y_mle_predict)}")
y_prob = mle_model.predict_proba(X_test)
print(f"Log loss : {log_loss(y_test,y_prob)}")

alphas = [0.1,1,10]

for a in alphas:
        map_model = make_pipeline(
            CountVectorizer(),
            MultinomialNB(alpha=a)
        )

        map_model.fit(X_train,y_train)
        y_map_predict = map_model.predict(X_test)
        print("MAP Metrics : ")
        print(f"Accuracy : {accuracy_score(y_test,y_map_predict)}")
        print(f"Log loss : {log_loss(y_test,y_prob)}")

print("Parameter Estimates : ")
print(f"MLE : {mle_model.named_steps['multinomialnb'].feature_log_prob_}")
print(f"MAP : {map_model.named_steps['multinomialnb'].feature_log_prob_}")
