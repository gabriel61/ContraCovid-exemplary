#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 08:59:34 2020

@author: Igor Dantas, Gabriel Sampaio
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numbers import Number
import sys

'''
Função para carregar e limpar os dados do dataset
'''

def load_dataset_sc (filename):

    # Lê o dataset com o pandas
    df = pd.read_csv(filename)
    
    # Substitui valores NaN por 0
    df = df.replace(np.nan, 0)
    # Substitui valores 'NAO INTERNADO UTI' por 0
    df = df.replace('NAO INTERNADO UTI', 0)
    # Substitui valores 'INTERNADO UTI' por 1
    df = df.replace('INTERNADO UTI', 1)
    # Substitui valores 'NAO INTERNADO' por 0
    df = df.replace('NAO INTERNADO', 0)
    # Substitui valores 'INTERNADO' por 1
    df = df.replace('INTERNADO', 1)
    # Substitui valores 'FEMININO' por 0
    df = df.replace('FEMININO', 0)
    # Substitui valores 'MASCULINO' por 1
    df = df.replace('MASCULINO', 1)
    
    # Seleciona as colunas relevantes para o modelo
    Xdf = df[[
    'sintomas',        
    'sexo',
    'idade',
    'comorbidades',
    'internacao',
    'internacao_uti',
    'obito',
    'tipo_teste'
    ]]
    
    # Lista com os nomes das features selecionadas
    feat_names = list(Xdf.columns)
    
    # Lista com os nomes reais das features
    real_feat_names = [
    'TOSSE',
    'DISPNEIA',
    'DOR NO CORPO',
    'DOR DE GARGANTA',
    'FEBRE',
    'CEFALEIA',
    'MIALGIA',
    'CORIZA',
    'DIARREIA',
    'CANSACO',
    'CONGESTAO NASAL',        
    'sexo',
    'idade',
    'comorbidades'
    ]
    
    # print(real_feat_names)

    # Matriz de features
    X = np.zeros((Xdf.shape[0], len(real_feat_names)))

    # Matriz de labels
    Y = np.zeros((Xdf.shape[0],1))

    # print(X.shape)
    
    # Loop para preencher a matriz de features
    for i in range (Xdf.shape[0]):
        # Converte para string a coluna de sintomas
        word = str(Xdf['sintomas'][i])
        # Loop para marcar as features presentes na string de sintomas
        for j in range (len(real_feat_names)-3):    # Itera sobre o range dos nomes de features, menos sexo, idade e comorbidades
            # Verifica se a feature está presente na string de sintomas
            if (word.find(real_feat_names[j]) != -1): 
                X[i,j] = 1  # Marca 1 na coluna correspondente
        X[i,11] = Xdf['sexo'][i]    # Adiciona o valor de sexo na coluna 11
        X[i,12] = Xdf['idade'][i]/100   # Adiciona o valor da idade na coluna 12
        # Verifica se há comorbidade
        if str(Xdf['comorbidades'][i])!="0":
            X[i,13] = 1 # marca 1 na coluna correspondente

        # Preenche a coluna de resposta "internacao"
        Y[i,0] = Xdf['internacao'][i]

    # Converte X e Y para float
    X = X.astype('float') 
    Y = Y.astype('float') 

    # Retorna X, Y e o nome das features reais do dataset
    return X, Y, real_feat_names