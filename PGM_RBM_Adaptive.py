# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 15:49:10 2024

@author: Lancelot Tullio
"""

#####################################
#Chargement des librairies utilisées#
#####################################

from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.single_table import CTGANSynthesizer
from sdv.single_table import TVAESynthesizer
from sdv.single_table import CopulaGANSynthesizer
from scipy.stats import entropy
from scipy.stats import kendalltau
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
import matplotlib.pyplot as plt
import statistics
from scipy.stats import norm
from sklearn.impute import SimpleImputer


############################################
#Importation de la base de données initiale#
############################################

DF = pd.read_csv('NAME_OF_DATAFRAME_PANDAS.csv', delimiter = ';', header=0)
DF.isna().sum()

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


##############################################################################
#Automatiser les simulations en fonction des 4 méthodologies autre que la RBM#
##############################################################################

class MesurePerformance:
    def __init__(self, df1, df2):
        self.df1 = df1
        self.df2 = df2
        
    def calculate_kendall_tau(self, var1, var2):
        tau, p_value = kendalltau(var1, var2)
        return tau

    def calculate_kendall_tau_for_variable(self, variable):
        if variable not in self.df1.columns or variable not in self.df2.columns:
            raise ValueError(f"La variable {variable} n'est pas présente dans les deux DataFrames.")       
        tau = self.calculate_kendall_tau(self.df1[variable], self.df2[variable])
        return tau

    def calculate_all_kendall_taux(self):
        common_variables = set(self.df1.columns) & set(self.df2.columns)
        results = []
        original_order = [col for col in self.df1.columns if col in common_variables]
        for variable in original_order:
            tau = self.calculate_kendall_tau_for_variable(variable)
            results.append({'Variable': variable, 'Kendall_Tau': tau})        
        result_df = pd.DataFrame(results)

        return result_df

    def ks_test(self):
        assert len(self.df1.columns) == len(self.df2.columns)
        result = pd.DataFrame(columns=['KS statistic', 'Décision'])
        for col in self.df1.columns:
            x = self.df1[col].values
            y = self.df2[col].values     
            n1 = len(x)
            n2 = len(y)
            all_values = np.sort(np.concatenate([x, y]))
            cdf1 = np.searchsorted(x, all_values, side='right') / n1
            cdf2 = np.searchsorted(y, all_values, side='right') / n2
            max_distance = np.mean(np.abs(cdf1 - cdf2))
            if max_distance > 0.01034:
                decision = 'Distributions différentes'
            elif 0.00942 <= max_distance < 0.01034:
                decision = 'Distributions légèrement différentes'
            elif 0.00692 <= max_distance < 0.00942:
                decision = 'Distributions similaires'
            else:
                decision = 'Distributions très similaires'
            result.loc[col] = [max_distance, decision]
        result.index.name = 'Variable'  
        return result


def kl_divergence(df1, df2):
    kl_div = []
    for col in df1.columns:
        if df1[col].dtype == 'object':
            # variable catégorielle
            p = df1[col].value_counts(normalize=True)
            q = df2[col].value_counts(normalize=True)
        else:
            # variable numérique
            bins = np.linspace(min(df1[col].min(), df2[col].min()), max(df1[col].max(), df2[col].max()), 10)
            p, _ = np.histogram(df1[col], bins=bins, density=True)
            q, _ = np.histogram(df2[col], bins=bins, density=True)
            # ajouter une constante pour éviter les valeurs nulles ==> Sinon cela engendre des valeurs infinie de la divergence de KL
            eps = 0.5 * np.min(p[p > 0])
            p = p + eps
            q = q + eps
            # normaliser les distributions de probabilité
            p = p / np.sum(p)
            q = q / np.sum(q)
        kl_div.append({'variable': col, 'KL_divergence': entropy(p, q)})
    return pd.DataFrame(kl_div)


####################################################
#Chargement des fonctions composant la RBM modifiée#
####################################################


 def type_variables(data):
     quali_var = data.select_dtypes(exclude=['float64', 'int64', 'float32', 'int32'])    
     for column in quali_var.columns:
         unique_values = quali_var[column].unique()
         mapping = {value: i for i, value in enumerate(unique_values)}
         data[column] = data[column].map(mapping)    
     num_var = data.select_dtypes(include=['float64', 'int64', 'float32', 'int32'])
     binaire_var = num_var.apply(pd.Series.nunique)==2
     binaire_var = binaire_var[binaire_var==True].index
     num_var = num_var.drop(binaire_var, axis=1)
     binaire_var = data[binaire_var].copy()  # Crée une copie explicite du DataFrame binaire_var
     for col in binaire_var.columns:
         if any(binaire_var.loc[:,col] == 2):
             binaire_var.loc[:, col] = binaire_var.loc[:,col].replace({2: 1, 1: 0})  
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

def Machine_Boltzmann_Adaptative(data):
  training_set_num, training_set_binaire = type_variables(data)
  data_initiale = pd.concat([training_set_num, training_set_binaire], axis=1)
  combinaison_types = data.dtypes.tolist()
  min_max_dict = {}
  for col in data_initiale.columns:
    min_max_dict[col] = {'min': data_initiale[col].min(), 'max': data_initiale[col].max()}
  training_set_num=normalisation(training_set_num)
  nb_client = len(data)
  nv=len(training_set_num.columns) + len(training_set_binaire.columns)
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
      for Gibbs in range(0,150):
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
#dt_genere = Machine_Boltzmann_Adaptative(DF)
#dt_1=dt_genere[0]
#dt_2=dt_genere[1]




#####################################################################
#Chargement des pré-requis pour réaliser les simulations avec la RBM#
#####################################################################

variable_names = DF.columns.tolist()
df_variable_names = pd.DataFrame({'Variables': variable_names})
folder_path = r"D:\Documents\Autre\These_RBM\Laboratoire\test"#/!\ Exemple de lien vers le dossier de sortie des résultats

accuracy_list_RBM = []

all_results_kendall_RBM = []
all_results_ks_RBM = []
kl_div_list_RBM=[]



for i in range(100):
    

    dt_genere = Machine_Boltzmann_Adaptative(DF)
        
    # Créez les paires de tables dt_1 et dt_2 pour cette itération
    dt_1 = dt_genere[0]
    dt_2_RBM = dt_genere[1]
        
    dt_1['y'] = 1
    dt_2_RBM['y'] = 0
    
    result_RBM = pd.concat([dt_1, dt_2_RBM], ignore_index=True)
    train_set_RBM, test_set_RBM = train_test_split(result_RBM, test_size=0.3, random_state=1)
    
    
    categorical_cols = [col for col in result_RBM.columns if result_RBM[col].dtype == 'object']
    numerical_cols = [col for col in result_RBM.columns if col not in categorical_cols and col != 'y']

    #Création d'un préprocesseur pour traiter les colonnes catégorielles et numériques séparément
    preprocessor = ColumnTransformer(
         transformers=[
             ('num', StandardScaler(), numerical_cols),  # traitement des variables numériques
             ('cat', OneHotEncoder(), categorical_cols)  # encodage one-hot des variables catégorielles
        ])

    #Application d'un contrôleur sous la forme d'une régression logistique pour limiter les hypothèses d'hyper-paramètres
    pipe = make_pipeline(preprocessor, LogisticRegression())
    pipe.fit(train_set_RBM.drop(columns=['y']), train_set_RBM['y'])
   
    y_test_pred_RBM = pipe.predict(test_set_RBM.drop(columns=['y']))

    accuracy_test_RBM = accuracy_score(test_set_RBM['y'], y_test_pred_RBM)
    print(f"Accuracy rate sur l'échantillon de test numéro {i} : {accuracy_test_RBM}")

    accuracy_list_RBM.append(accuracy_test_RBM)
    
    mesure_performance = MesurePerformance(dt_1, dt_2_RBM)
    
    result_kendall_RBM = mesure_performance.calculate_all_kendall_taux()
    result_ks_RBM = mesure_performance.ks_test()
    
    all_results_kendall_RBM.append(result_kendall_RBM)
    all_results_ks_RBM.append(result_ks_RBM)
    
    kl_div_RBM = kl_divergence(dt_1.drop(columns=['y']), dt_2_RBM.drop(columns=['y']))
    kl_div_list_RBM.append(kl_div_RBM)
   
 
############################################
#Exportation des résultats de niveau global#
############################################

accuracy_mean_RBM = np.mean(accuracy_list_RBM)
print(accuracy_mean_RBM)
accuracy_std_RBM = np.std(accuracy_list_RBM)
print(accuracy_std_RBM)


###############################################
#Exportation des résultats de niveau variables#
###############################################


df_final_KS = pd.DataFrame()

for i, df in enumerate(all_results_ks_RBM):
    df_final_KS[f'DataFrame_{i+1}'] = df.iloc[:, 0]  

mean_per_row = df_final_KS.mean(axis=1)
median_per_row = df_final_KS.median(axis=1)
std_per_row = df_final_KS.std(axis=1)
summary_df_KS = pd.DataFrame({
    'Mean': mean_per_row,
    'Median': median_per_row,
    'Std Deviation': std_per_row
})

summary_df_KS = pd.concat([summary_df_KS, df_variable_names], axis=1)


df_final_KEN = pd.DataFrame()

for i, df in enumerate(all_results_kendall_RBM):
    df_final_KEN[f'DataFrame_{i+1}'] = df.iloc[:, 1]  

mean_per_row = df_final_KEN.mean(axis=1)
median_per_row = df_final_KEN.median(axis=1)
std_per_row = df_final_KEN.std(axis=1)
summary_df_KEN = pd.DataFrame({
    'Mean': mean_per_row,
    'Median': median_per_row,
    'Std Deviation': std_per_row
})

summary_df_KEN = pd.concat([summary_df_KEN, df_variable_names], axis=1)


df_final_KL = pd.DataFrame()

for i, df in enumerate(kl_div_list_RBM):
    df_final_KL[f'DataFrame_{i+1}'] = df.iloc[:, 1]  

mean_per_row = df_final_KL.mean(axis=1)
median_per_row = df_final_KL.median(axis=1)
std_per_row = df_final_KL.std(axis=1)
summary_df_KL = pd.DataFrame({
    'Mean': mean_per_row,
    'Median': median_per_row,
    'Std Deviation': std_per_row
})

summary_df_KL = pd.concat([summary_df_KL, df_variable_names], axis=1)


excel_path = os.path.join(folder_path, "summary_df_KS_RBM.xlsx")
summary_df_KS.to_excel(excel_path, index=False)
excel_path = os.path.join(folder_path, "summary_df_KEN_RBM.xlsx")
summary_df_KEN.to_excel(excel_path, index=False)
excel_path = os.path.join(folder_path, "summary_df_KL_RBM.xlsx")
summary_df_KL.to_excel(excel_path, index=False)


























