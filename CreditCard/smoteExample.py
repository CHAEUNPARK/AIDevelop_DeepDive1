from CreditCard import creditcard
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings("ignore")

def logisticR(X_train, y_train, X_test, y_test, disc):
    lr = LogisticRegression()
    lr.fit(X_train, y_train.ravel())   #resample 전 모델 학습
    y_test_pre = lr.predict(X_test)

    print(disc + "accuracy_score    :{:.2f}%".format(accuracy_score(y_test, y_test_pre) * 100))
    print(disc + "recall_score      :{:.2f}%".format(recall_score(y_test, y_test_pre) * 100))
    print(disc + "precision_score   :{:.2f}%".format(precision_score(y_test, y_test_pre) * 100))
    print(disc + "roc_auc_score     :{:.2f}%".format(roc_auc_score(y_test, y_test_pre) * 100))

def rf(X_train, y_train, X_test, y_test, disc):
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train.ravel()) #resample한 모델
    y_test_pre = rf.predict(X_test)

    cnf_matrix_rf = confusion_matrix(y_test, y_test_pre)
    print(disc + "matrix_accuracy_score : ", (cnf_matrix_rf[1, 1] + cnf_matrix_rf[0, 0]) /
          (cnf_matrix_rf[1, 0] + cnf_matrix_rf[1, 1] + cnf_matrix_rf[0, 1] + cnf_matrix_rf[0, 0]) * 100)

    print(disc + "matrix_recall_score : ", (cnf_matrix_rf[1, 1] / (cnf_matrix_rf[1, 0] + cnf_matrix_rf[1, 1]) * 100))

if __name__ == "__main__":
    X_train = creditcard.X_train
    y_train = creditcard.y_train
    X_test = creditcard.X_test
    y_test = creditcard.y_test

    X_smote = creditcard.X_train_res
    y_smote = creditcard.y_train_res

    logisticR(X_train, y_train, X_test, y_test, "smote전+logisticR")
    logisticR(X_smote, y_smote, X_test, y_test, "smote후+logisticR")

    rf(X_train, y_train, X_test, y_test, "smote전+RF")
    rf(X_smote, y_smote, X_test, y_test, "smote후+RF")