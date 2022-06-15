import numpy as np
import time
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from csv import reader

# CSV файл
def load_csv(filename):
	file = open(filename, "r")
	lines = reader(file)
	dataset = list(lines)
	return dataset

filename = 'C:\\ffff\\attacking.csv'
dataset = load_csv(filename)
DATA =np.matrix(dataset)
output = open('results.txt', 'a')

# количество данных для обучения
num_train = 6
# счётчик неправильно предсказанных значений 
counter = 1000
# начало отсчёта
t_beg = time.time()
while counter > 0:
    # для обучения
    X_train = np.array(DATA[1:num_train, [4, 5, 6, 7]])
    Y_train = np.array(dataset)[1:num_train, 8]
    # для проверки
    X_test = np.array(DATA[num_train:, [4, 5, 6, 7]])
    Y_test = np.array(DATA[num_train:, 8])
    # классификатор
    clf = svm.SVC()
    # обучение классификатора
    clf.fit(X_train, Y_train)

    # предсказываем количество матчей
    predicted = clf.predict(X_test)
    # счётчик для неверно предсказанных результатов
    counter = 0
    for i in range(0, len(predicted)):
        #print("for assists = ", X_test[i][0], ", corner_taken = ", X_test[i][1]," offsides = ", X_test[i][2], "dribbles =", X_test[i][3] " - predicted number of match played is ", predicted[i])
        if Y_test[i] != predicted[i]:
            counter += 1
    num_train += 1

output.write("Amount of training data is " + str(num_train) + "   Number of mispredicted results is " + str(counter)+ '\n')
# окончание отсчёта времени 
t_end = time.time()
t = t_end - t_beg
output.write("With SVM it took " + str(t) + " seconds " + '\n')



# количество данных для обучения
num_train = 6
# счётчик неправильно предсказанных значений 
counter = 1000
# начало отсчёта
t_beg = time.time()
while counter > 0:
    # для обучения 
    X_train = np.array(DATA[1:num_train, [4, 5, 6, 7]], dtype=int)
    Y_train = np.array(DATA[1:num_train, 8], dtype=int)
    # для проверки
    X_test = np.array(DATA[num_train:, [4, 5, 6,7 ]],dtype=int)
    Y_test = np.array(DATA[num_train:, 8], dtype=int)

    # классификатор
    neigh = KNeighborsClassifier()
    # обучение классификаторв
    neigh.fit(X_train, Y_train.ravel())

    # счётчик для неверно предсказанных результатов
    counter = 0
    # предсказание количества матчей
    for i in range(0, len(X_test)):
        predicted = neigh.predict([X_test[i]])
        if Y_test[i] != predicted:
            counter += 1
    num_train += 1

output.write("Amount of training data is " + str(num_train) + "   Number of mispredicted results is " + str(counter)+ '\n')
# окончание отсчёта времени 
t_end = time.time()
t = t_end - t_beg
output.write("With kNN it took " + str(t) + " seconds " + '\n')
