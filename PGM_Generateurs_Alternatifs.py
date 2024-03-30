# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 14:59:12 2024

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
import pandas as pd
import numpy as np
from scipy.stats import kendalltau
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


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



##########################################################################
#Chargement des pré-requis pour réaliser les simulations des 4 challenger#
##########################################################################

variable_names = DF.columns.tolist()
df_variable_names = pd.DataFrame({'Variables': variable_names})
folder_path = r"D:\Documents\Autre\These_RBM\Laboratoire\test"#/!\ Exemple de lien vers le dossier de sortie des résultats

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=DF)

Nb_simulations=DF.shape[0]

synthesizer_CG = GaussianCopulaSynthesizer(metadata)
synthesizer_CG.fit(DF)

synthesizer_CTGAN = CTGANSynthesizer(metadata)
synthesizer_CTGAN.fit(DF)

synthesizer_TVAE = TVAESynthesizer(metadata)
synthesizer_TVAE.fit(DF)

synthesizer_COPGAN = CopulaGANSynthesizer(metadata)
synthesizer_COPGAN.fit(DF)

accuracy_list_CG = []
accuracy_list_CTGAN = []
accuracy_list_TVAE= []
accuracy_list_COPGAN= []

all_results_kendall_CG = []
all_results_ks_CG = []
kl_div_list_CG=[]

all_results_kendall_CTGAN = []
all_results_ks_CTGAN = []
kl_div_list_CTGAN=[]

all_results_kendall_TVAE = []
all_results_ks_TVAE = []
kl_div_list_TVAE=[]

all_results_kendall_COPGAN = []
all_results_ks_COPGAN = []
kl_div_list_COPGAN=[]


for i in range(100):
    
    dt_1 = DF
    dt_1['y'] = 1
    
    ########################################
    #Simulateur à base de Copule Gaussienne#
    ########################################

    dt_2_CG = synthesizer_CG.sample(num_rows=Nb_simulations)
    
    dt_2_CG['y'] = 0
    
    result_CG = pd.concat([dt_1, dt_2_CG], ignore_index=True)
    train_set_CG, test_set_CG = train_test_split(result_CG, test_size=0.3, random_state=1)
    
    
    categorical_cols = [col for col in result_CG.columns if result_CG[col].dtype == 'object']
    numerical_cols = [col for col in result_CG.columns if col not in categorical_cols and col != 'y']

    #Création d'un préprocesseur pour traiter les colonnes catégorielles et numériques séparément
    preprocessor = ColumnTransformer(
         transformers=[
             ('num', StandardScaler(), numerical_cols),  #traitement des variables numériques
             ('cat', OneHotEncoder(), categorical_cols)  #encodage one-hot des variables catégorielles
        ])

    #Application d'un contrôleur sous la forme d'une régression logistique pour limiter les hypothèses d'hyper-paramètres
    pipe = make_pipeline(preprocessor, LogisticRegression())
    pipe.fit(train_set_CG.drop(columns=['y']), train_set_CG['y'])
   
    y_test_pred_CG = pipe.predict(test_set_CG.drop(columns=['y']))

    accuracy_test_CG = accuracy_score(test_set_CG['y'], y_test_pred_CG)
    print(f"Accuracy rate sur l'échantillon de test numéro {i} : {accuracy_test_CG}")

    accuracy_list_CG.append(accuracy_test_CG)
    
    mesure_performance = MesurePerformance(dt_1, dt_2_CG)
    
    result_kendall_CG = mesure_performance.calculate_all_kendall_taux()
    result_ks_CG = mesure_performance.ks_test()
    
    all_results_kendall_CG.append(result_kendall_CG)
    all_results_ks_CG.append(result_ks_CG)
    
    kl_div_CG = kl_divergence(dt_1.drop(columns=['y']), dt_2_CG.drop(columns=['y']))
    kl_div_list_CG.append(kl_div_CG)
   
    #############################
    #Simulateur à base de CT GAN#
    #############################

    dt_2_CTGAN = synthesizer_CTGAN.sample(num_rows=Nb_simulations)
    dt_2_CTGAN['y'] = 0

    result_CTGAN = pd.concat([dt_1, dt_2_CTGAN], ignore_index=True)
    train_set_CTGAN, test_set_CTGAN = train_test_split(result_CTGAN, test_size=0.3, random_state=1)

    categorical_cols = [col for col in result_CTGAN.columns if result_CTGAN[col].dtype == 'object']
    numerical_cols = [col for col in result_CTGAN.columns if col not in categorical_cols and col != 'y']

    #Création d'un préprocesseur pour traiter les colonnes catégorielles et numériques séparément
    preprocessor = ColumnTransformer(
         transformers=[
             ('num', StandardScaler(), numerical_cols),  #traitement des variables numériques
             ('cat', OneHotEncoder(), categorical_cols)  #encodage one-hot des variables catégorielles
        ])

    #Application d'un contrôleur sous la forme d'une régression logistique pour limiter les hypothèses d'hyper-paramètres
    pipe = make_pipeline(preprocessor, LogisticRegression())
    pipe.fit(train_set_CTGAN.drop(columns=['y']), train_set_CTGAN['y'])
   
    y_test_pred_CTGAN = pipe.predict(test_set_CTGAN.drop(columns=['y']))

    accuracy_test_CTGAN = accuracy_score(test_set_CTGAN['y'], y_test_pred_CTGAN)
    print(f"Accuracy rate sur l'échantillon de test numéro {i} : {accuracy_test_CTGAN}")

    accuracy_list_CTGAN.append(accuracy_test_CTGAN)

    mesure_performance = MesurePerformance(dt_1, dt_2_CTGAN)
    
    result_kendall_CTGAN = mesure_performance.calculate_all_kendall_taux()
    result_ks_CTGAN = mesure_performance.ks_test()
    
    all_results_kendall_CTGAN.append(result_kendall_CTGAN)
    all_results_ks_CTGAN.append(result_ks_CTGAN)
    
    kl_div_CTGAN = kl_divergence(dt_1.drop(columns=['y']), dt_2_CTGAN.drop(columns=['y']))
    kl_div_list_CTGAN.append(kl_div_CTGAN)

   
    ###########################
    #Simulateur à base de TVAE#
    ###########################

    dt_2_TVAE = synthesizer_TVAE.sample(num_rows=Nb_simulations)
    
    dt_2_TVAE['y'] = 0
    
    result_TVAE = pd.concat([dt_1, dt_2_TVAE], ignore_index=True)
    train_set_TVAE, test_set_TVAE = train_test_split(result_TVAE, test_size=0.3, random_state=1)
    
    
    categorical_cols = [col for col in result_TVAE.columns if result_TVAE[col].dtype == 'object']
    numerical_cols = [col for col in result_TVAE.columns if col not in categorical_cols and col != 'y']
    
    #Création d'un préprocesseur pour traiter les colonnes catégorielles et numériques séparément
    preprocessor = ColumnTransformer(
         transformers=[
             ('num', StandardScaler(), numerical_cols),  #traitement des variables numériques
             ('cat', OneHotEncoder(), categorical_cols)  #encodage one-hot des variables catégorielles
        ])
    
    #Application d'un contrôleur sous la forme d'une régression logistique pour limiter les hypothèses d'hyper-paramètres
    pipe = make_pipeline(preprocessor, LogisticRegression())
    pipe.fit(train_set_TVAE.drop(columns=['y']), train_set_TVAE['y'])
    
    y_test_pred_TVAE = pipe.predict(test_set_TVAE.drop(columns=['y']))
    
    accuracy_test_TVAE = accuracy_score(test_set_TVAE['y'], y_test_pred_TVAE)
    print(f"Accuracy rate sur l'échantillon de test numéro {i} : {accuracy_test_TVAE}")
    
    accuracy_list_TVAE.append(accuracy_test_TVAE)
    
    mesure_performance = MesurePerformance(dt_1, dt_2_TVAE)
    
    result_kendall_TVAE = mesure_performance.calculate_all_kendall_taux()
    result_ks_TVAE= mesure_performance.ks_test()
    
    all_results_kendall_TVAE.append(result_kendall_TVAE)
    all_results_ks_TVAE.append(result_ks_TVAE)
    
    kl_div_TVAE = kl_divergence(dt_1.drop(columns=['y']), dt_2_TVAE.drop(columns=['y']))
    kl_div_list_TVAE.append(kl_div_TVAE)


    ############################################
    #Simulateur à base d'une copule et d'un GAN#
    ############################################

    dt_2_COPGAN = synthesizer_COPGAN.sample(num_rows=Nb_simulations)
    
    dt_2_COPGAN['y'] = 0
    
    result_COPGAN = pd.concat([dt_1, dt_2_COPGAN], ignore_index=True)
    train_set_COPGAN, test_set_COPGAN = train_test_split(result_COPGAN, test_size=0.3, random_state=1)
    
    
    categorical_cols = [col for col in result_COPGAN.columns if result_COPGAN[col].dtype == 'object']
    numerical_cols = [col for col in result_COPGAN.columns if col not in categorical_cols and col != 'y']
    
    #Création d'un préprocesseur pour traiter les colonnes catégorielles et numériques séparément
    preprocessor = ColumnTransformer(
         transformers=[
             ('num', StandardScaler(), numerical_cols),  #traitement des variables numériques
             ('cat', OneHotEncoder(), categorical_cols)  #encodage one-hot des variables catégorielles
        ])
    
    #Application d'un contrôleur sous la forme d'une régression logistique pour limiter les hypothèses d'hyper-paramètres
    pipe = make_pipeline(preprocessor, LogisticRegression())
    pipe.fit(train_set_COPGAN.drop(columns=['y']), train_set_COPGAN['y'])
    
    y_test_pred_COPGAN = pipe.predict(test_set_COPGAN.drop(columns=['y']))
    
    accuracy_test_COPGAN = accuracy_score(test_set_COPGAN['y'], y_test_pred_COPGAN)
    print(f"Accuracy rate sur l'échantillon de test numéro {i} : {accuracy_test_COPGAN}")
    
    accuracy_list_COPGAN.append(accuracy_test_COPGAN)
    
    mesure_performance = MesurePerformance(dt_1, dt_2_COPGAN)
    
    result_kendall_COPGAN = mesure_performance.calculate_all_kendall_taux()
    result_ks_COPGAN = mesure_performance.ks_test()
    
    all_results_kendall_COPGAN.append(result_kendall_COPGAN)
    all_results_ks_COPGAN.append(result_ks_COPGAN)
    
    kl_div_COPGAN = kl_divergence(dt_1.drop(columns=['y']), dt_2_COPGAN.drop(columns=['y']))
    kl_div_list_COPGAN.append(kl_div_COPGAN)


############################################
#Exportation des résultats de niveau global#
############################################

accuracy_mean_CG = np.mean(accuracy_list_CG)
print(accuracy_mean_CG)
accuracy_std_CG = np.std(accuracy_list_CG)
print(accuracy_std_CG)

accuracy_mean_CTGAN = np.mean(accuracy_list_CTGAN)
print(accuracy_mean_CTGAN)
accuracy_std_CTGAN = np.std(accuracy_list_CTGAN)
print(accuracy_std_CTGAN)

accuracy_mean_TVAE = np.mean(accuracy_list_TVAE)
print(accuracy_mean_TVAE)
accuracy_std_TVAE= np.std(accuracy_list_TVAE)
print(accuracy_std_TVAE)

accuracy_mean_COPGAN = np.mean(accuracy_list_COPGAN)
print(accuracy_mean_COPGAN)
accuracy_std_COPGAN = np.std(accuracy_list_COPGAN)
print(accuracy_std_COPGAN)



###############################################
#Exportation des résultats de niveau variables#
###############################################



    ########################################
    #Simulateur à base de Copule Gaussienne#
    ########################################

df_final_KS = pd.DataFrame()

for i, df in enumerate(all_results_ks_CG):
    df_final_KS[f'DataFrame_{i+1}'] = df.iloc[:, 1]  

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

for i, df in enumerate(all_results_kendall_CG):
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

for i, df in enumerate(kl_div_list_CG):
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


excel_path = os.path.join(folder_path, "summary_df_KS_CG.xlsx")
summary_df_KS.to_excel(excel_path, index=False)
excel_path = os.path.join(folder_path, "summary_df_KEN_CG.xlsx")
summary_df_KEN.to_excel(excel_path, index=False)
excel_path = os.path.join(folder_path, "summary_df_KL_CG.xlsx")
summary_df_KL.to_excel(excel_path, index=False)


    #############################
    #Simulateur à base de CT GAN#
    #############################


df_final_KS_v1 = pd.DataFrame()

for i, df in enumerate(all_results_ks_CTGAN):
    df_final_KS_v1[f'DataFrame_{i+1}'] = df.iloc[:, 1]  

mean_per_row = df_final_KS_v1.mean(axis=1)
median_per_row = df_final_KS_v1.median(axis=1)
std_per_row = df_final_KS_v1.std(axis=1)
summary_df_KS_v1 = pd.DataFrame({
    'Mean': mean_per_row,
    'Median': median_per_row,
    'Std Deviation': std_per_row
})

summary_df_KS_v1 = pd.concat([summary_df_KS_v1, df_variable_names], axis=1)


df_final_KEN_v1 = pd.DataFrame()

for i, df in enumerate(all_results_kendall_CTGAN):
    df_final_KEN_v1[f'DataFrame_{i+1}'] = df.iloc[:, 1]  

mean_per_row = df_final_KEN_v1.mean(axis=1)
median_per_row = df_final_KEN_v1.median(axis=1)
std_per_row = df_final_KEN_v1.std(axis=1)
summary_df_KEN_v1 = pd.DataFrame({
    'Mean': mean_per_row,
    'Median': median_per_row,
    'Std Deviation': std_per_row
})

summary_df_KEN_v1 = pd.concat([summary_df_KEN_v1, df_variable_names], axis=1)


df_final_KL_v1 = pd.DataFrame()

for i, df in enumerate(kl_div_list_CTGAN):
    df_final_KL_v1[f'DataFrame_{i+1}'] = df.iloc[:, 1]  

mean_per_row = df_final_KL_v1.mean(axis=1)
median_per_row = df_final_KL_v1.median(axis=1)
std_per_row = df_final_KL_v1.std(axis=1)
summary_df_KL_v1 = pd.DataFrame({
    'Mean': mean_per_row,
    'Median': median_per_row,
    'Std Deviation': std_per_row
})

summary_df_KL_v1 = pd.concat([summary_df_KL_v1, df_variable_names], axis=1)


excel_path = os.path.join(folder_path, "summary_df_KS_CTGAN.xlsx")
summary_df_KS_v1.to_excel(excel_path, index=False)
excel_path = os.path.join(folder_path, "summary_df_KEN_CTGAN.xlsx")
summary_df_KEN_v1.to_excel(excel_path, index=False)
excel_path = os.path.join(folder_path, "summary_df_KL_CTGAN.xlsx")
summary_df_KL_v1.to_excel(excel_path, index=False)



    ###########################
    #Simulateur à base de TVAE#
    ###########################


df_final_KS_v2 = pd.DataFrame()

for i, df in enumerate(all_results_ks_TVAE):
    df_final_KS_v2[f'DataFrame_{i+1}'] = df.iloc[:, 1]  

mean_per_row = df_final_KS_v2.mean(axis=1)
median_per_row = df_final_KS_v2.median(axis=1)
std_per_row = df_final_KS_v2.std(axis=1)
summary_df_KS_v2 = pd.DataFrame({
    'Mean': mean_per_row,
    'Median': median_per_row,
    'Std Deviation': std_per_row
})

summary_df_KS_v2 = pd.concat([summary_df_KS_v2, df_variable_names], axis=1)


df_final_KEN_v2 = pd.DataFrame()

for i, df in enumerate(all_results_kendall_TVAE):
    df_final_KEN_v2[f'DataFrame_{i+1}'] = df.iloc[:, 1]  

mean_per_row = df_final_KEN_v2.mean(axis=1)
median_per_row = df_final_KEN_v2.median(axis=1)
std_per_row = df_final_KEN_v2.std(axis=1)
summary_df_KEN_v2 = pd.DataFrame({
    'Mean': mean_per_row,
    'Median': median_per_row,
    'Std Deviation': std_per_row
})

summary_df_KEN_v2 = pd.concat([summary_df_KEN_v2, df_variable_names], axis=1)


df_final_KL_v2 = pd.DataFrame()

for i, df in enumerate(kl_div_list_TVAE):
    df_final_KL_v2[f'DataFrame_{i+1}'] = df.iloc[:, 1]  

mean_per_row = df_final_KL_v2.mean(axis=1)
median_per_row = df_final_KL_v2.median(axis=1)
std_per_row = df_final_KL_v2.std(axis=1)
summary_df_KL_v2 = pd.DataFrame({
    'Mean': mean_per_row,
    'Median': median_per_row,
    'Std Deviation': std_per_row
})

summary_df_KL_v2 = pd.concat([summary_df_KL_v2, df_variable_names], axis=1)


excel_path = os.path.join(folder_path, "summary_df_KS_TVAE.xlsx")
summary_df_KS_v2.to_excel(excel_path, index=False)
excel_path = os.path.join(folder_path, "summary_df_KEN_TVAE.xlsx")
summary_df_KEN_v2.to_excel(excel_path, index=False)
excel_path = os.path.join(folder_path, "summary_df_KL_TVAE.xlsx")
summary_df_KL_v2.to_excel(excel_path, index=False)


    ############################################
    #Simulateur à base d'une copule et d'un GAN#
    ############################################

df_final_KS_v3 = pd.DataFrame()

for i, df in enumerate(all_results_ks_COPGAN):
    df_final_KS_v3[f'DataFrame_{i+1}'] = df.iloc[:, 1]  

mean_per_row = df_final_KS_v3.mean(axis=1)
median_per_row = df_final_KS_v3.median(axis=1)
std_per_row = df_final_KS_v3.std(axis=1)
summary_df_KS_v3 = pd.DataFrame({
    'Mean': mean_per_row,
    'Median': median_per_row,
    'Std Deviation': std_per_row
})

summary_df_KS_v3 = pd.concat([summary_df_KS_v3, df_variable_names], axis=1)


df_final_KEN_v3 = pd.DataFrame()

for i, df in enumerate(all_results_kendall_COPGAN):
    df_final_KEN_v3[f'DataFrame_{i+1}'] = df.iloc[:, 1]  

mean_per_row = df_final_KEN_v3.mean(axis=1)
median_per_row = df_final_KEN_v3.median(axis=1)
std_per_row = df_final_KEN_v3.std(axis=1)
summary_df_KEN_v3 = pd.DataFrame({
    'Mean': mean_per_row,
    'Median': median_per_row,
    'Std Deviation': std_per_row
})

summary_df_KEN_v3 = pd.concat([summary_df_KEN_v3, df_variable_names], axis=1)


df_final_KL_v3 = pd.DataFrame()

for i, df in enumerate(kl_div_list_COPGAN):
    df_final_KL_v3[f'DataFrame_{i+1}'] = df.iloc[:, 1]  

mean_per_row = df_final_KL_v3.mean(axis=1)
median_per_row = df_final_KL_v3.median(axis=1)
std_per_row = df_final_KL_v3.std(axis=1)
summary_df_KL_v3 = pd.DataFrame({
    'Mean': mean_per_row,
    'Median': median_per_row,
    'Std Deviation': std_per_row
})

summary_df_KL_v3 = pd.concat([summary_df_KL_v3, df_variable_names], axis=1)


excel_path = os.path.join(folder_path, "summary_df_KS_COPGAN.xlsx")
summary_df_KS_v3.to_excel(excel_path, index=False)
excel_path = os.path.join(folder_path, "summary_df_KEN_COPGAN.xlsx")
summary_df_KEN_v3.to_excel(excel_path, index=False)
excel_path = os.path.join(folder_path, "summary_df_KL_COPGAN.xlsx")
summary_df_KL_v3.to_excel(excel_path, index=False)












