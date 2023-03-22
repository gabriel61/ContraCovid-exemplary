#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 2 11:37:48 2020

@authors: Gabriel Sampaio
"""

'''
Esta função recebe uma matriz de confusão como entrada e calcula várias métricas de avaliação de desempenho do modelo de classificação.
'''

def assess_classifier (confusion_mat):

    # Extrai os valores da matriz de confusão
    tn = confusion_mat[0,0] # verdadeiros negativos (true negatives)
    fp = confusion_mat[0,1] # falsos positivos (false positives)
    fn = confusion_mat[1,0] # falsos negativos (false negatives)
    tp = confusion_mat[1,1] # verdadeiros positivos (true positives)

    # Cálculo da taxa de falsos positivos
    fpr = 1 - tn/(tn + fp)
    
    # Cálculo da taxa de falsos negativos
    fnr = 1 - tp/(tp + fn)

    # A precisão (precision) é a proporção de verdadeiros positivos em relação ao total de positivos previstos
    precision = tp/(tp + fp)
    
    # O recall é a proporção de verdadeiros positivos em relação ao total de positivos reais. 
    # Ou seja, é a proporção de verdadeiros positivos que foram identificados corretamente.
    recall = tp/(tp + fn)

    # O F1 é uma métrica que combina precisão e recall, é usada para buscar um equilíbrio entre ambas
    f1 = 2*(precision*recall)/(precision+recall)
    
    # A acurácia (accuracy) é a proporção de predições corretas em relação ao total de predições
    # Ou seja, diz quanto o meu modelo acertou das previsões possíveis.
    acc = (tp+tn)/(tp+tn+fp+fn)
    
    # Cria um dicionário com as métricas calculadas e retorna o resultado
    scores = {}
    scores["FPR"] = fpr
    scores["FNR"] = fnr
    scores["F1"] = f1 
    scores["ACC"] = acc
    
    return scores