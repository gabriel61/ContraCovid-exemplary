#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 12:10:55 2020

@author: Igor Dantas

"""
import numpy as np

'''
Esta função foi desenvolvida para limitar o tamanho do conjunto de dados de um problema de classificação binária.
A função seleciona um número máximo de amostras de cada classe (positiva e negativa) para serem incluídas no conjunto de dados final.
O tratamento de limitação é necessário para o conjunto de dados usado neste script, pois há mais pacientes não hospitalizados do que hospitalizados.
'''

def limit_dataset (X_a, Y_a, max_ones=1000000, max_zeros=1000000, shuffle=1):
    
    """
    Essa função recebe como entrada uma matriz X_a com dimensão (n,m), representando um conjunto de dados de n amostras e m atributos,
    e um vetor de rótulos Y_a com dimensão (n,1), onde cada elemento representa o rótulo (0 ou 1) da respectiva amostra em X_a.
    A função também recebe argumentos opcionais max_ones, max_zeros e shuffle.
    A função retorna uma matriz X_vec com dimensão (n',m) e um vetor de rótulos Y_vec com dimensão (n',1),
    onde n' é o número de amostras selecionadas para serem incluídas no conjunto de dados final.
    
    A função utiliza as seguintes regras para selecionar as amostras que serão incluídas no conjunto de dados final:
    - Se o rótulo da amostra for 1 (positivo) e o número máximo de amostras positivas (max_ones) não tiver sido alcançado,
    então a amostra é incluída no conjunto de dados final.
    - Se o rótulo da amostra for 0 (negativo) e o número máximo de amostras negativas (max_zeros) não tiver sido alcançado,
    então a amostra é incluída no conjunto de dados final.
    - Se o número máximo de amostras positivas ou negativas já tiver sido alcançado, a amostra é ignorada.

    Se o argumento shuffle for 1 (valor padrão), as amostras serão embaralhadas aleatoriamente antes de serem selecionadas.
    """

    # Obtém o número de amostras em X_a
    n_vecs = X_a.shape[0]

    # Embaralha aleatoriamente as amostras em X_a e Y_a se o argumento shuffle for 1
    if(shuffle==1):
        shuf_ind = np.arange(n_vecs)
        np.random.shuffle(shuf_ind)
        X = X_a[shuf_ind,:]
        Y = Y_a[shuf_ind,:]
    else:
        X = X_a
        Y = Y_a
        
    # Inicializa os vetores de rótulos e atributos do conjunto de dados final
    Y_vec = np.array([], dtype=np.int64).reshape(0,Y.shape[1])
    X_vec = np.array([], dtype=np.int64).reshape(0,X.shape[1])

    # Inicializa as contagens de amostras positivas e negativas
    cnt_ones = 0
    cnt_zeros = 0
    
    # Seleciona amostras para o conjunto de dados final seguindo as regras especificadas
    # Percorre todas as amostras do conjunto de dados original
    for i in range(Y.shape[0]):
        # Se o rótulo da amostra for 1 (positivo) e o número máximo de amostras positivas (max_ones) não tiver sido alcançado,
        # ou se o rótulo da amostra for 0 (negativo) e o número máximo de amostras negativas (max_zeros) não tiver sido alcançado,
        # a amostra é incluída no conjunto de dados final.
        if((Y[i,0]==1 and cnt_ones<max_ones) or (Y[i,0]==0 and cnt_zeros<max_zeros)):
            Y_vec = np.vstack([Y_vec, Y[i,0]])
            X_vec = np.vstack([X_vec, X[i,:]])
            # Incrementa o contador correspondente ao número de amostras selecionadas da classe da amostra atual
            if(Y[i,0]==1):
                cnt_ones+=1
            else:
                cnt_zeros+=1
            
    # Retorna o conjunto de dados final, com o número máximo de amostras selecionadas de cada classe
    return X_vec, Y_vec