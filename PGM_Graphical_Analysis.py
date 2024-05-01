# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 16:44:12 2024

@author: Lancelot Tullio
"""

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.impute import SimpleImputer
import numpy as np
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer

############################################
#Importation de la base de données initiale#
############################################

DF = pd.read_csv('wine.csv', delimiter = ';', header=0)
DF.isna().sum()

DF2 = pd.read_csv('FED_ECO.csv', delimiter = ';', header=0)
DF2.isna().sum()

#####################################################
#Traitement des valeurs manquantes avant utilisation#
#####################################################

qualitative_cols = DF.select_dtypes(include=['object']).columns
quantitative_cols = DF.select_dtypes(exclude=['object']).columns
for col in qualitative_cols:
    DF[col].fillna('VM', inplace=True)
imputer = SimpleImputer(strategy='mean')
for col in quantitative_cols:
    DF[col] = imputer.fit_transform(DF[[col]])


#####################################################
#Réalisation des graphiques 3D sur les deux datasets#
#####################################################

def Graph_3d_densite_surface(df, var1, var2, var3):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    x = df[var1].values
    y = df[var2].values
    z = df[var3].values
    
    #Créer une grille 3D
    X, Y = np.meshgrid(np.linspace(min(x), max(x), 100), np.linspace(min(y), max(y), 100))
    Z = np.zeros_like(X)
    
    #Interpoler les valeurs de Z
    for i in range(len(x)):
        Z += (x[i] - X) ** 2 + (y[i] - Y) ** 2 + (z[i] - np.mean(z)) ** 2
    
    #Tracer la surface
    #ax.plot_surface(X, Y, Z, cmap='cool')
    #version FED_ECO
    ax.plot_surface(X, Y, Z, cmap='plasma_r')
    
    ax.set_xlabel(var1)
    ax.set_ylabel(var2)
    ax.set_zlabel(var3)
    ax.set_title(f'Surface 3D en fonction de {var1}, {var2} et {var3}')
    
    plt.show()

#Générer le graphique des données réelles
Graph_3d_densite_surface(DF, 'Alcohol', 'Ash','Proline')
Graph_3d_densite_surface(DF,'Proanthocyanins','Hue','Color_intensity')

Graph_3d_densite_surface(DF2,'Taux_Pop_Active_USA','Taux_Chomage_USA','Nb_Employee_USA')
Graph_3d_densite_surface(DF2,'Rendement_OAT_3ans_USA','Masse_Monetaire_M1','Taux_Pop_Active_USA')


#Générer des données avec la RBM -- Charger le programme contenant la Machine de Boltzmann avant utilisation
dt_genere = Machine_Boltzmann_Adaptative(DF)
dt_1=dt_genere[0]

dt_genere = Machine_Boltzmann_Adaptative(DF2)
dt_1_bis=dt_genere[0]

#Générer le graphique des données générées
Graph_3d_densite_surface(dt_1, 'Alcohol', 'Ash','Proline')
Graph_3d_densite_surface(dt_1, 'Proanthocyanins','Hue','Color_intensity')

Graph_3d_densite_surface(dt_1_bis, 'Taux_Pop_Active_USA','Taux_Chomage_USA','Nb_Employee_USA')
Graph_3d_densite_surface(dt_1_bis, 'Rendement_OAT_3ans_USA','Masse_Monetaire_M1','Taux_Pop_Active_USA')


#Générer des données avec la Copule Gaussienne
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=DF)
synthesizer_CG = GaussianCopulaSynthesizer(metadata)
synthesizer_CG.fit(DF)
Nb_simulations=DF.shape[0]
dt_1_CG = synthesizer_CG.sample(num_rows=Nb_simulations)

#Générer le graphique des données générées
Graph_3d_densite_surface(dt_1_CG, 'Alcohol', 'Ash','Proline')
Graph_3d_densite_surface(dt_1_CG, 'Proanthocyanins','Hue','Color_intensity')


metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=DF2)
synthesizer_CG2 = GaussianCopulaSynthesizer(metadata)
synthesizer_CG2.fit(DF2)
Nb_simulations=DF2.shape[0]
dt_1_CG2 = synthesizer_CG2.sample(num_rows=Nb_simulations)

Graph_3d_densite_surface(dt_1_CG2, 'Taux_Pop_Active_USA','Taux_Chomage_USA','Nb_Employee_USA')
Graph_3d_densite_surface(dt_1_CG2,'Rendement_OAT_3ans_USA','Masse_Monetaire_M1','Taux_Pop_Active_USA')





