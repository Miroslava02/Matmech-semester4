import numpy as np
from sklearn.cluster import KMeans
from csv import reader
import matplotlib.pyplot as plt
# загружаем CSV файл
def load_csv(filename):
	file = open(filename, "r")
	lines = reader(file)
	dataset = list(lines)
	return dataset
filename = 'C:\\ffff\ds_salaries.csv'
dataset = load_csv(filename)
DATA = np.matrix(dataset)
output = open('results.txt', 'a')

num_train=100
# первые num_train результатов используем для обучения
X_train=np.array(DATA[1:num_train, [1, 7]],int)
# остальные результаты используем для проверки
X_test=np.array(DATA[num_train:, [1, 7]], int)
# проводим кластеризацию
kmeans = KMeans().fit(X_train)
# массив, показывающий к какому кластеру принадлежит каждый элемент X_train
labels = kmeans.labels_
# создаем массивы для разных кластеров
X_green = np.empty((0,2))
X_blue = np.empty((0,2))
X_cyan = np.empty((0,2))
X_purple = np.empty((0,2))
X_yellow = np.empty((0,2))
X_lime = np.empty((0,2))
X_orange = np.empty((0,2))
X_pink = np.empty((0,2))
for i in range(0, len(X_train)):
    if labels[i] == 0:
        X_green = np.append(X_green, np.array([X_train[i]]), axis=0)
    elif labels[i] == 1:
        X_blue = np.append(X_blue, np.array([X_train[i]]), axis=0)
    elif labels[i] == 2:
        X_cyan = np.append(X_cyan, np.array([X_train[i]]), axis=0)
    elif labels[i] == 3:
        X_purple = np.append(X_purple, np.array([X_train[i]]), axis=0)
    elif labels[i] == 4:
        X_yellow = np.append(X_yellow, np.array([X_train[i]]), axis=0)
    elif labels[i] == 5:
        X_lime = np.append(X_lime, np.array([X_train[i]]), axis=0)
    elif labels[i] == 6:
        X_orange = np.append(X_orange, np.array([X_train[i]]), axis=0)
    else:
        X_pink = np.append(X_pink, np.array([X_train[i]]), axis=0)
# создаём окно с двумя графиками 
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.canvas.set_window_title('Сluster analysis results')
ax1.set_title('Test data')
ax2.set_title('Train data')
ax1.set_xlim(2019,2022)
ax1.set_ylim(10000,500000)
ax2.set_xlim(2019,2022)
ax2.set_ylim(10000,500000)
# точки разных кластеров рисуем разными цветами 
ax1.scatter(x=X_green[:,0], y=X_green[:,1], marker='o', c='g')
ax1.scatter(x=X_blue[:,0], y=X_blue[:,1], marker='o', c='b')
ax1.scatter(x=X_cyan[:,0], y=X_cyan[:,1], marker='o', c='c')
ax1.scatter(x=X_purple[:,0], y=X_purple[:,1], marker='o', c='m')
ax1.scatter(x=X_yellow[:,0], y=X_yellow[:,1], marker='o', c='yellow')
ax1.scatter(x=X_lime[:,0], y=X_lime[:,1], marker='o', c='lime')
ax1.scatter(x=X_orange[:,0], y=X_orange[:,1], marker='o', c='darkorange')
ax1.scatter(x=X_pink[:,0], y=X_pink[:,1], marker='o', c='pink')
# рисуем центры кластеров
centers = kmeans.cluster_centers_
ax1.scatter(centers[:,0], centers[:,1 ], marker = 'D', c='r', edgecolor='k')
# предсказываем, в какие кластеры попадают тестовые значения 
pred_labels=kmeans.predict(X_test)
print(pred_labels)
for i in range(0, len(X_test)):
	if pred_labels[i] == 0:
		color = 'g'
	elif pred_labels[i] == 1:
		color = 'b'
	elif pred_labels[i] == 2:
		color = 'c'
	elif pred_labels[i] == 3:
		color = 'm'
	elif pred_labels[i] == 4:
		color = 'yellow'
	elif pred_labels[i] == 5:
		color = 'lime'
	elif pred_labels[i] == 6:
		color = 'darkorange'
	else:
		color = 'pink'
	ax2.scatter(X_test[i,0], X_test[i,1 ], marker = 'o', c=color)

plt.show() 
