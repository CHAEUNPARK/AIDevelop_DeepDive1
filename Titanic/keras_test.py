from Titanic import train as preTrain
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import pandas as pd
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense

# Function to create model, required for KerasClassifier
def create_model(optimizer='adam', init='glorot_uniform'):
    model = Sequential()
    model.add(Dense(10, input_dim=8, kernel_initializer=init, activation='relu'))
    model.add(Dense(15))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
    return model

train = preTrain.train
test = preTrain.test

Y_train_data = pd.DataFrame(train['Survived'].values.reshape(-1,1), columns=['Survived'])
X_train_data = train.drop(['Survived'], axis=1)

X_train = X_train_data
y_train = Y_train_data

model = KerasClassifier(build_fn=create_model, verbose=0)
# model = create_model()
# optimizers = ['rmsprop', 'adam']
# init = ['glorot_uniform', 'normal', 'uniform']
epochs = [50, 100]
batches = [3, 5]
param_grid = dict(epochs=epochs,
                  batch_size=batches)

gs = GridSearchCV(estimator=model,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=5,
                  n_jobs=-1)

gs.fit(X_train, y_train)

print(gs.best_params_)
print(gs.best_score_)

# model.fit(X_train, y_train, epochs=300, batch_size=3)
#
# loss, acc = model.evaluate(X_train, y_train, batch_size=3)
#
# print("acc : ", acc)

# Todo: grid search 너무 오래 걸림 해결 방안 찾기