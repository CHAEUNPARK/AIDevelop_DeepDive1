from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from IPython.display import display
import pandas as pd
from sklearn.model_selection import cross_validate

def train_split():
    x, y = make_blobs(random_state=0)

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

    logreg = LogisticRegression().fit(x_train, y_train)
    print("테스트 세트 점수 : {:.2f}".format(logreg.score(x_test, y_test)))

def k_fold():
    iris = load_iris()
    kf_data = iris.data
    kf_label = iris.target
    kf_columns = iris.feature_names

    rf = RandomForestClassifier(n_estimators=100,
                                max_depth=10,
                                random_state=2019)
    scores = cross_val_score(rf, kf_data, kf_label, cv=10)
    print(scores)
    print('rf k-fold CV score:{:.2f}%'.format(scores.mean()))

def k_fold_validate():
    iris = load_iris()
    kf_data = iris.data
    kf_label = iris.target
    kf_columns = iris.feature_names

    rf = RandomForestClassifier(n_estimators=100,
                                max_depth=10,
                                random_state=2019)
    scores = cross_validate(rf, kf_data, kf_label, cv=10, return_train_score=True)
    print("<< score >>")
    print(scores)
    res_df = pd.DataFrame(scores)
    print("<< res_df >>")
    display(res_df)
    print('평균 시간과 점수 : \n',res_df)

if __name__ =="__main__":
    train_split()
    k_fold()
    k_fold_validate()