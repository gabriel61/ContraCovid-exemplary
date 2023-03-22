    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 2 11:37:48 2020

@authors: Gabriel Sampaio
"""

''' 
Este projeto tem o intuito de prever se um paciente vai ou não ser hospitalizado.
Usamos como base um dataset de Santa Catarina com cerca de 200 mil pacientes,
todos positivos para Covid-19. Essa base de dados também foi usada em outro
projeto (ContraCovid - Projeto ministrado pelo professor Dr. Igor Dantas Dos Santos Miranda) 
onde pude fazer parte como membro da equipe.

A técnica utilizada no projeto original foi a regressão logística, então, para o projeto
da turma de IA, ministrada pela professora Camila Bezerra durante o semestre
suplementar de 2020.3, irei usar a técnica de Árvore de Decisão (Decission Tree), 
aprendida durante o curso de IA.
'''


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from load_dataset_sc import load_dataset_sc
from sklearn.model_selection import train_test_split
from limit_dataset import limit_dataset
from assess_classifier import assess_classifier

'''
Este script implementa a classificação de pacientes para prever se um paciente vai ou não ser
hospitalizado com base em seus sintomas, idade, comorbidades e sexo usando a técnica de árvore de decisão.
O conjunto de dados usado neste script é baseado em pacientes com Covid-19 em Santa Catarina e foi dividido em
partes de treino e teste usando validação cruzada.
'''

'''
Após testes, avaliamos que, por conta do dataset ser muito grande, ele acabava ficando
"viciado". Por esse motivo, dividimos o dataset usando a técnica de CROSS VALIDATION,
dividindo e treinando N partes diferentes do dataset.
'''

# Definindo número de partes para o Cross Validation e semente para reprodutibilidade
N_CROSS_VALIDATION = 10
seed = 10

# Carrega os dados fazendo as limpezas necessárias
[Xr,Yr,feat_names] = load_dataset_sc('boavista_covid_dados_abertos_reduzida.csv')

'''
A variável global acc_global serve para armazenar a acurácia total das N vezes em que o Cross Validation
executará os processos de treino, teste e predição, a fim de calcular, no final, a média das predições em cada parte.
'''
acc_global  = 0

# Executando o Cross Validation
for n_cross in range(N_CROSS_VALIDATION):
    
    # Limita o número de casos negativos (dataset unbalanced)
    # Definimos o número de casos positivos e usamos a função limit_dataset para limitar o número de casos negativos
    pos_cases = np.sum(Yr)
    [X, Y] = limit_dataset (Xr, Yr, max_ones=pos_cases,max_zeros=pos_cases, shuffle=1)
    
    # Separa o dataset em parte para treino (25%) e outra para teste (75%) (validação)
    # Usando train_test_split para separar o dataset em dados de treino e teste
    # test_size é definido como 0.25, então 75% dos dados são usados para treino e 25% para teste
    # stratify é definido como Y, então a proporção de cada classe é mantida nas partes de treino e teste
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, stratify=Y, random_state=1)

    # Aqui estamos criando um objeto de Árvore de Decisão com os hiperparâmetros definidos
    # criterion é definido como 'gini', que é um critério para medir a qualidade das divisões
    # min_samples_leaf é definido como 5, que é o número mínimo de amostras permitido em uma folha
    # min_samples_split é definido como 5, que é o número mínimo de amostras requerido para dividir um nó
    # max_depth é definido como None, o que significa que não há limite para a profundidade da árvore
    # random_state é definido como seed, que é uma semente para o gerador de números aleatórios
    tree = DecisionTreeClassifier(criterion='gini',
                                  min_samples_leaf=5,
                                  min_samples_split=5,
                                  max_depth=None,
                                  random_state=seed)
    
    # Usamos a função fit() do objeto tree para treinar a árvore de decisão com os dados de treino
    tree.fit(x_train, y_train)
    
    # Usamos a função predict() do objeto tree para prever a classe dos dados de teste
    y_pred = tree.predict(x_test)
    
    # Matrix de confusão para comparar o encontrado vs esperado
    # Usando a função confusion_matrix para criar a matriz de confusão
    # y_test é a classe real dos dados de teste e y_pred é a classe prevista pela árvore de decisão
    # A matriz de confusão é uma tabela usada para avaliar o desempenho do modelo
    predictions = np.array(y_pred)
    cm = confusion_matrix(y_test, predictions)
    print(cm)
    
    # Usando a função assess_classifier para calcular várias métricas de avaliação a partir da matriz de confusão
    scores = assess_classifier (cm)
    print(scores)
    
    # Adicionando a acurácia ao contador acc_global
    acc_global += scores['ACC']

# Por fim, tiramos a média da Acurácia
acc_global /= N_CROSS_VALIDATION

print("\n\n>>> Global Accuraccy: %.04f" % (acc_global))

'''
Após executar o cross validation e somar as acurácias obtidas em cada fold, 
o código tira a média dessas acurácias para obter a acurácia global do modelo. 
Em seguida, o resultado é impresso na tela com duas casas decimais de precisão. 
Finalmente, há um comentário que resume os resultados obtidos, indicando uma acurácia de 88% nas predições
de pacientes que serão hospitalizados com base em seus sintomas, idade, comorbidades e sexo. 
O resultado é considerado excelente e ficou apenas 1% abaixo do resultado do projeto original que utilizou regressão logística.
'''