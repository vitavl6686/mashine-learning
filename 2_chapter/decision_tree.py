import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def main():
	new_data = pd.read_csv('titanic.csv', index_col = 'PassengerId')

	#Оставим в выборке только нужные нам признаки
	test_data = new_data.filter(items = ['Pclass', 'Fare', 'Age', 'Sex', 'Survived'])
	#выкинем объекты с недостаточными данными
	without_NaN = test_data.dropna()
	#Пол -- непонятная срока. Заменяем его на бин.характеристику
	_valid_test_data = without_NaN.replace('male', 1).replace('female',0)

	valid_test_data = _valid_test_data.filter(items = ['Pclass', 'Fare', 'Age', 'Sex'])

	answers = _valid_test_data.filter(items = ['Survived'])

	clf = DecisionTreeClassifier()
	#Обучаем решающее дерево на нашей выборке
	clf.fit(valid_test_data,answers)
	#Посмотрим наши клевые важнсти признаков
	importances = clf.feature_importances_
	#print(importances)
if __name__ == '__main__':
	main()
