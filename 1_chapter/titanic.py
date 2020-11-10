import pandas as pd
new_data = pd.read_csv('titanic.csv', index_col = 'PassengerId')

###########Количество мужчин и женщин на корабле###########
sexs = new_data['Sex'].value_counts()
#print(sexs)

###########Доля выживших############
survived = new_data['Survived'].value_counts(normalize = 'True')
#print(survived)

##########Доля пассажиров первого класса###########
the_first_class = new_data['Pclass'].value_counts(normalize = 'True')
#print(the_first_class)

##########Среднее и медиана возраста пассажиров##########
ages = new_data['Age']
#print(ages.mean())
#print(ages.median())

##########Коррелируют ли число братьев/сестер/супругов с числом родителей/детей?###########
corr = new_data['SibSp'].corr(new_data['Parch'])
#print(corr)
