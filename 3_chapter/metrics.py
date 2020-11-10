import pandas as pd
import sklearn
from sklearn import model_selection
from sklearn import neighbors, preprocessing
data = pd.read_csv('wine.data')

#Вытаскиваем тестовые признаки
X_columns = data.columns[1:]
X_test = data[X_columns]

#Выстаскиваем ответы
Y_columns = data.columns[0]
Y_test = data[Y_columns]

#Создаем генератор разбиений
generator = sklearn.model_selection.KFold(n_splits = 5, shuffle = True, random_state = 42)
list_of_accuracy = list() #список аккуратности на разном количестве соседей
list_of_accuracy.append(0) #потому что метода 0 соседей нет:))
for i in range(1,51):
	classifier = sklearn.neighbors.KNeighborsClassifier(n_neighbors = i)
	#эта штука выдает массив аккуратностей на 5 разбиениях
	accuracy = sklearn.model_selection.cross_val_score(estimator = classifier, cv = generator, X = X_test, y = Y_test, scoring = 'accuracy')
	#а тут мы считаем среднюю аккуратность
	mean_accuracy = accuracy.mean()
	list_of_accuracy.append(mean_accuracy)
#вывод максимальной аккуратности
#print(max(list_of_accuracy))
#вывод того, при каком она кол-ве соседей
#print(list_of_accuracy.index(max(list_of_accuracy)))



########Масштабируем признаки
scaled_X = sklearn.preprocessing.scale(X_test)
list_of_accuracy = list() #список аккуратности на разном количестве соседей
list_of_accuracy.append(0) #потому что метода 0 соседей нет:))
for i in range(1,51):
	classifier = sklearn.neighbors.KNeighborsClassifier(n_neighbors = i)
	#эта штука выдает массив аккуратностей на 5 разбиениях
	accuracy = sklearn.model_selection.cross_val_score(estimator = classifier, cv = generator, X = scaled_X, y = Y_test, scoring = 'accuracy')
	#а тут мы считаем среднюю аккуратность
	mean_accuracy = accuracy.mean()
	list_of_accuracy.append(mean_accuracy)
#вывод максимальной аккуратности
#print(max(list_of_accuracy))
#вывод того, при каком она кол-ве соседей
#print(list_of_accuracy.index(max(list_of_accuracy)))