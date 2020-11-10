import numpy as np

#########создадим новую матрицу с рандомными числами из нормального распределения N(1,100)

new_matrix = np.random.normal(loc = 50, scale = 1, size = (2,5))

#########найдем матрицу целых чисел
int_matrix = np.random.randint(low = 0, high = 10, size = (2,5))
#print(int_matrix)
#########найдем среднее значение в столбце
mean = np.mean(int_matrix,axis = 0)
#print(mean)

#########вычтем среднее значение столбца из каждого столбца
half_norm_matrix = int_matrix - mean
#print(half_norm_matrix) 

#########найдем стандартное отклонение в каждом столбце
std = np.std(int_matrix, axis = 0)
#print(std)

#########отнормируем матрицу
normal_matrix = (int_matrix - mean)/std
#print(normal_matrix)

#########подсчитаем сумму строк матрицы
sum_matrix = np.sum(int_matrix, axis = 1)
#print(np.nonzero(sum_matrix > 10))


###ЕДИНИЧНЫЕ МАТРИЦЫ и вертикальная стыковка
a = np.eye(3)
b = np.eye(3)
ab = np.vstack((a,b))

cat = np.random.randint(low = 0, high = 100, size = (3,3))
dog = np.random.randint(low = 0, high = 100, size = (4,3))
#print(cat)
#print(dog)
#print(np.vstack((cat,dog)))
