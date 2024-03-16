# -*- coding: utf-8 -*-
"""
@author: Lancelot Tullio
"""

#Déclaration des librairies 
import pandas as pd
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
import matplotlib.pyplot as plt
import statistics
from scipy.stats import norm
from sklearn.impute import SimpleImputer

#Importation de la base de données en format PANDAS
DF = pd.read_csv('Data_Bank.csv', delimiter = ';', header=0)
#Vérifier la présence de Missing values 
DF.isna().sum()


#En cas de présence des Missing values le code ci-dessous propose une imputation par la moyenne pour les quanti et la création d'une nouvelle modalité pour les variables qualitatives
qualitative_cols = DF.select_dtypes(include=['object']).columns
quantitative_cols = DF.select_dtypes(exclude=['object']).columns
for col in qualitative_cols:
    DF[col].fillna('VM', inplace=True)
imputer = SimpleImputer(strategy='mean')
for col in quantitative_cols:
    DF[col] = imputer.fit_transform(DF[[col]])


#Définition des fonctions appelées dans la A_RBM

def type_variables(data):
    quali_var = data.select_dtypes(exclude=['float64', 'int64', 'float32', 'int32'])    
    for column in quali_var.columns:
        unique_values = quali_var[column].unique()
        mapping = {value: i for i, value in enumerate(unique_values)}
        data[column] = data[column].map(mapping)    
    num_var = data.select_dtypes(include=['float64', 'int64', 'float32', 'int32'])
    binaire_var = num_var.apply(pd.Series.nunique)==2
    binaire_var = binaire_var[binaire_var==True].index
    num_var=num_var.drop(binaire_var, axis=1)
    binaire_var = data[binaire_var]  
    for col in binaire_var.columns:
        if any(binaire_var[col] == 2):
            binaire_var[col] = binaire_var[col].replace({2: 1, 1: 0})   
    return num_var, binaire_var

def normalisation(data):
    data_bis=data.copy()
    for colonne in data.columns:
        minimum=data[colonne].min()
        maximum=data[colonne].max()
        data_bis[colonne]=(data[colonne]-minimum)/(maximum - minimum)
    return(data_bis)

def convert(data):
    new_data = []
    nb=len(data)
    for observation in range(0, nb):
        value = []
        for col in data.columns:
            value.append(data[col].values[observation])
        new_data.append(value)
    return new_data

def hist_dataframe(data, bins=50, color='#00008B', edgecolor='#00008B'):
    n = len(data.columns)
    rows = int(np.ceil(n / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(8, 4*rows))
    axes = axes.ravel()
    for i, col in enumerate(data.columns):
        ax = axes[i]
        ax.hist(data[col], bins=bins, color=color, edgecolor=edgecolor)
        ax.set_ylabel(col)
    plt.tight_layout()

def denormalisation(data, dt_initial):
    data_bis=data.copy()
    for colonne in data.columns:
        minimum=dt_initial[colonne].min()
        maximum=dt_initial[colonne].max()
        data_bis[colonne]=data[colonne]*(maximum - minimum) + minimum
    return(data_bis)

#Définition de la Machine de Boltzmann Adaptative : 

def Machine_Boltzmann_Adaptative(data, Gibbs=150):
  training_set_num, training_set_binaire = type_variables(data)
  data_initiale = pd.concat([training_set_num, training_set_binaire], axis=1)
  combinaison_types = data.dtypes.tolist()
  min_max_dict = {}
  for col in data_initiale.columns:
    min_max_dict[col] = {'min': data_initiale[col].min(), 'max': data_initiale[col].max()}
  training_set_num=normalisation(training_set_num)
  nb_client = len(data)
  nv=len(training_set_num.columns) + len(training_set_binaire.columns)
  #Définition du nombre de neurones cachés ==> potentiellement à mettre en hyper-paramètre
  nh=round((2/3)*nv)  
  training_set = pd.concat([training_set_num, training_set_binaire], axis=1)  
  new_data=None
  vec_mean=None 
  a = torch.randn(1, nh)
  b = torch.randn(1, nv)
  W = torch.randn(nh, nv)
  nb_variables_num = len(training_set_num.columns)
  for variable in range(0,nb_variables_num):
      if vec_mean is None:
          vec_mean=[statistics.mean(training_set_num.iloc[variable])]
          vec_sd=[statistics.stdev(training_set_num.iloc[variable])]
      else:
          bis=statistics.mean(training_set_num.iloc[variable])
          bis_sd=statistics.stdev(training_set_num.iloc[variable])
          vec_mean.append(bis)
          vec_sd.append(bis_sd)    
  for observation in range(0, nb_client):     
      if observation % 500 == 0:
         print(f"Avancement : {observation}/{nb_client} observations traitées.")      
      Gibbs=0
      dt=training_set.iloc[observation:1+observation]
      dt = convert(dt)
      dt_tensor = torch.FloatTensor(dt)
      for Gibbs in range(0,Gibbs):
          mat2 = dt_tensor
          wx=torch.mm(mat2, W.t())
          activation = wx + a.expand_as(wx)
          p_h_given_v = torch.sigmoid(activation)        
          ph0=torch.bernoulli(p_h_given_v)
          y = ph0 
          wy=torch.mm(y, W)
          activation = wy + b
          p_h_given_h = torch.sigmoid(activation)
          Taille_globale=len(training_set.columns)
          Taille_Data_Binaire=len(training_set_binaire.columns)
          diff_taille=Taille_globale - Taille_Data_Binaire
          p_h_given_h_num = p_h_given_h.narrow(1, 0, diff_taille)
          p_h_given_h_binaire = p_h_given_h.narrow(1, diff_taille, Taille_Data_Binaire)
          result_num=norm.ppf(p_h_given_h_num , loc=vec_mean, scale=vec_sd)
          result_binaire = torch.bernoulli(p_h_given_h_binaire)
          result_binaire=result_binaire.numpy()
          result = np.concatenate((result_num, result_binaire), axis=1)
          result = torch.FloatTensor(result)         
          Valeur_result=torch.mean(result[~torch.isinf(result)]).item()
          result[torch.isinf(result)] = Valeur_result  
          W += (torch.mm(dt_tensor.t(), p_h_given_v) - torch.mm(result.t(), ph0)).t()
          b += dt_tensor - result
          a += p_h_given_v - ph0         
      result_final=result
      result_final=result_final.tolist()
      Nom_colonnes = training_set.columns.tolist()
      result_final=pd.DataFrame(result_final,columns=Nom_colonnes)
      if new_data is None:
         new_data=result_final
      else:
         new_data=pd.concat([new_data,result_final]).reset_index(drop=True)  
  new_data = new_data[data.columns]
  data_initiale = data_initiale[data.columns]
  dt_final=denormalisation(new_data, data)
  for i, column in enumerate(dt_final.columns):
    type_actuel = dt_final[column].dtype
    type_initial = combinaison_types[i]
    if type_actuel != type_initial:
        if type_actuel == float and type_initial == int:
            dt_final[column] = dt_final[column].round().astype(int)
        elif type_actuel == float:
            dt_final[column] = dt_final[column].round(3)
        else:
            dt_final[column] = dt_final[column].astype(type_initial)
  for col in dt_final.columns:
      min_val = min_max_dict[col]['min']
      max_val = min_max_dict[col]['max'] 
      dt_final[col] = dt_final[col].apply(lambda x: min_val if x < min_val else x)
      dt_final[col] = dt_final[col].apply(lambda x: max_val if x > max_val else x)
  #hist_dataframe(data_initiale)
  #hist_dataframe(dt_final)  
  return dt_final, data_initiale

#Appel de la Machine
dt_genere = Machine_Boltzmann_Adaptative(DF)
dt_1=dt_genere[0]
dt_2=dt_genere[1]
