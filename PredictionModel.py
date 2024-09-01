import pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

#Displaying Data
data_set = pandas.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv')

#Deleting every missing value (NaN) because it will mess up the model's predictions
data_set.dropna(axis=0, how='any', subset=None, inplace=True )
data_set.tail(10)

#%%
#One hot encode non_numerics
data_set = pandas.get_dummies(data_set, columns=['island','sex'], dtype=int)
data_set.tail()
#%%
#Assign X and y variables
X = data_set.drop('species', axis=1)
y = data_set['species']

#Split training vs testing data (70/30)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, shuffle=True)
#%%
#Create Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

#%%
#Evaluate model
model_test = model.predict(X_test)

#Confusion matrix test
print(confusion_matrix(y_test, model_test))
#Classification report
print(classification_report(y_test, model_test))
#We are comparing the actual species of penguins from the original data set to the model's prediction of the species using confusion matrix and classification report


#%%
#Test Case
penguin = pandas.DataFrame([[
   47.2,
    13.7,
    214.0,
    4925.0,
    1,
    0,
    0,
    1,
    0
]], columns=X.columns)
model_prediction = model.predict(penguin)
print(f'Based on the model prediction, the penguin species is {model_prediction[0]}')
