import sklearn
import numpy as np
from sklearn import datasets, preprocessing, model_selection,neighbors

def main():
	data = sklearn.datasets.load_boston()
	X = data.data
	Y_test = data.target
	#масштабируем признаки обучающей выборки
	X_test = sklearn.preprocessing.scale(X)
	#200 вариантов параметра метрики миньковского
	array = np.linspace(start = 1, stop = 10, num = 200)
	#создаем генератор разбиений
	generator = sklearn.model_selection.KFold(n_splits = 5, shuffle = True, random_state = 42)
	list_of_accuracy = list()
	list_of_p = list()
	for i in array:
		#собствевнно, регрессор
		regressor = sklearn.neighbors.KNeighborsRegressor(n_neighbors = 5, weights = 'distance', p = i)
		accuracy = sklearn.model_selection.cross_val_score(estimator = regressor, X = X_test, y = Y_test, cv = generator, scoring='neg_mean_squared_error')
		mean_accuracy = accuracy.mean()
		list_of_accuracy.append(mean_accuracy)
		list_of_p.append(i)
	max_accuracy = max(list_of_accuracy)
	index_of = list_of_accuracy.index(max_accuracy)
	the_best_p = list_of_p[index_of]
	#print(round(the_best_p,2))

if __name__=='__main__':
	main()