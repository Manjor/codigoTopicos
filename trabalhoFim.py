# Importanto as libs
print('Importando as Libs...\n')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print('Carregando base de dados...\n')
# Carregar base de dados
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784')
print('Base de dados carregada.')

# Criando as bases de treino e teste
print('Fazendo Treinamento...\n')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    mnist.data, mnist.target, test_size=1/7.0, random_state=0)
print('Fim do Treinamento.\n')


posicao = 40

def gerarRL():
    print('Gerando Regressão Logistica...')
    from sklearn.linear_model import LogisticRegression
    cl = LogisticRegression(solver = 'lbfgs')
    cl.fit(X_train, y_train)
    plt.imshow(np.reshape(X_test[posicao], (28,28)))
    prev = cl.predict(X_test[posicao].reshape(1,-1))
    print('Previsão: ' + str(prev))
    return cl

def gerarKNN():
    print('Gerando KNN...')
    from sklearn.neighbors import KNeighborsClassifier
    cl = KNeighborsClassifier(metric='minkowski')
    cl.fit(X_train, y_train)
    plt.imshow(np.reshape(X_test[posicao], (28,28)))
    prev = cl.predict(X_test[posicao].reshape(1,-1))
    print('Previsão: ' + str(prev))
    return cl

def gerarSVM():
    print('Gerando SVM RBF...')
    from sklearn.svm import SVC
    cl = SVC(kernel = 'rbf', random_state = 0 )
    cl.fit(X_train, y_train)
    plt.imshow(np.reshape(X_test[posicao], (28,28)))
    prev = cl.predict(X_test[posicao].reshape(1,-1))
    print('Previsão: ' + str(prev))
    return cl

def gerarSVMpoly():
    print('Gerando SVM Poly...')
    from sklearn.svm import SVC
    cl = SVC(kernel = 'poly', random_state = 0 )
    cl.fit(X_train, y_train)
    plt.imshow(np.reshape(X_test[posicao], (28,28)))
    prev = cl.predict(X_test[posicao].reshape(1,-1))
    print('Previsão: ' + str(prev))
    return cl

def gerarClassificao(cls):
    if cls == '1':
        return gerarRL()
    elif cls == '2':
        return gerarKNN()
    elif cls == '3':
        return gerarSVM()
    elif cls == '4':
        return gerarSVMpoly()
    else:
        return 0


def gerarFigura(clsFigure,X_test, y_test):
    from sklearn.metrics import plot_confusion_matrix
    plt.rcParams['figure.figsize'] = [12, 12]
    plot_confusion_matrix(clsFigure, X_test, y_test, normalize='true')
    plt.show()

def verificarAcuracia(teste, pred):
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(teste, pred)
    print('Acurácia: ' + str(acc))

indices = ['0','1','2','3','4','5','6','7','8','9']

classificador = input('Digite o numero do classificador desejado:\n' +
'1 - Regressão Logistica\n' +
'2 - KNN\n' + 
'3 - SVM(rbf)\n' +
'4 - SVM(poly)')

classificacao = gerarClassificao(classificador)
tempo = 0
if classificacao == 0:
    print('Nenhuma opção válida foi inserida.')
else:
    import time
    inicio = time.time()
    y_pred = classificacao.predict(X_test)
    fim = time.time()
    tempo = round(fim - inicio, 2)
    print('Tempo das previsões: ' + str(tempo) + ' segundos')
    from sklearn.metrics import confusion_matrix
    matrizCon = confusion_matrix(y_test, y_pred)
    print(matrizCon)
    matrizData = pd.DataFrame(matrizCon, index = indices,
                  columns = indices)
    gerarFigura(classificacao,X_test, y_test)
    verificarAcuracia(y_test, y_pred)
