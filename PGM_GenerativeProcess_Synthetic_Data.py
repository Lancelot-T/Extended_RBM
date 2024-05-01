# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 09:50:09 2024

@author: Lancelot Tullio
"""


from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer
import pandas as pd
import numpy as np
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

DF = pd.read_csv('AutoMPG.csv', delimiter = ';', header=0)
DF.isna().sum()

DF2 = pd.read_csv('Breast.csv', delimiter = ';', header=0)
DF2.isna().sum()

DF3 = pd.read_csv('Cancer.csv', delimiter = ';', header=0)
DF3.isna().sum()

DF4 = pd.read_csv('Ecoli.csv', delimiter = ';', header=0)
DF4.isna().sum()

DF5 = pd.read_csv('FED_ECO.csv', delimiter = ';', header=0)
DF5.isna().sum()

DF6 = pd.read_csv('Heart.csv', delimiter = ';', header=0)
DF6.isna().sum()

DF7 = pd.read_csv('iris.csv', delimiter = ';', header=0)
DF7.isna().sum()

DF8 = pd.read_csv('StatLog.csv', delimiter = ';', header=0)
DF8.isna().sum()

DF9 = pd.read_csv('Student.csv', delimiter = ';', header=0)
DF9.isna().sum()

DF10 = pd.read_csv('wine.csv', delimiter = ';', header=0)
DF10.isna().sum()

#####################################################
#Traitement des valeurs manquantes avant utilisation#
#####################################################

def Traitement_VM(df):
    qualitative_cols = df.select_dtypes(include=['object']).columns
    quantitative_cols = df.select_dtypes(exclude=['object']).columns
    for col in qualitative_cols:
        df[col].fillna('VM', inplace=True)
    imputer = SimpleImputer(strategy='mean')
    for col in quantitative_cols:
        df[col] = imputer.fit_transform(df[[col]])
    return df

DF=Traitement_VM(DF)
DF.isna().sum()

DF2=Traitement_VM(DF2)
DF2.isna().sum()

DF6=Traitement_VM(DF6)
DF6.isna().sum()

DF8=Traitement_VM(DF8)
DF8.isna().sum()


##################################################################
#Générer des données synthétiques pour chaque dataset avec la RBM#
##################################################################

# /!\ lancer le programme contenant la RBM avant execution des lignes de code

folder_path = r"D:\Documents\Autre\These_RBM\Laboratoire\Data_Base_Small\Data_Generees\RBM"

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

dt_genere = Machine_Boltzmann_Adaptative(DF)
dt_AutoMPG=dt_genere[0]
excel_path = os.path.join(folder_path, "AutoMPG_Generative.xlsx")
dt_AutoMPG.to_excel(excel_path, index=False)

dt_genere = Machine_Boltzmann_Adaptative(DF2)
dt_Breast=dt_genere[0]
excel_path = os.path.join(folder_path, "Breast_Generative.xlsx")
dt_Breast.to_excel(excel_path, index=False)

dt_genere = Machine_Boltzmann_Adaptative(DF3)
dt_Cancer=dt_genere[0]
excel_path = os.path.join(folder_path, "Cancer_Generative.xlsx")
dt_Cancer.to_excel(excel_path, index=False)

dt_genere = Machine_Boltzmann_Adaptative(DF4)
dt_Ecoli=dt_genere[0]
excel_path = os.path.join(folder_path, "Ecoli_Generative.xlsx")
dt_Ecoli.to_excel(excel_path, index=False)

dt_genere = Machine_Boltzmann_Adaptative(DF5)
dt_FED_ECO=dt_genere[0]
excel_path = os.path.join(folder_path, "FED_ECO_Generative.xlsx")
dt_FED_ECO.to_excel(excel_path, index=False)

dt_genere = Machine_Boltzmann_Adaptative(DF6)
dt_Heart=dt_genere[0]
excel_path = os.path.join(folder_path, "Heart_Generative.xlsx")
dt_Heart.to_excel(excel_path, index=False)

dt_genere = Machine_Boltzmann_Adaptative(DF7)
dt_iris=dt_genere[0]
excel_path = os.path.join(folder_path, "iris_Generative.xlsx")
dt_iris.to_excel(excel_path, index=False)

dt_genere = Machine_Boltzmann_Adaptative(DF8)
dt_StatLog=dt_genere[0]
excel_path = os.path.join(folder_path, "StatLog_Generative.xlsx")
dt_StatLog.to_excel(excel_path, index=False)

dt_genere = Machine_Boltzmann_Adaptative(DF9)
dt_Student=dt_genere[0]
excel_path = os.path.join(folder_path, "Student_Generative.xlsx")
dt_Student.to_excel(excel_path, index=False)

dt_genere = Machine_Boltzmann_Adaptative(DF10)
dt_wine=dt_genere[0]
excel_path = os.path.join(folder_path, "wine_Generative.xlsx")
dt_wine.to_excel(excel_path, index=False)


################################################################################
#Générer des données synthétiques pour chaque dataset avec la Copule Gaussienne#
################################################################################


folder_path = r"D:\Documents\Autre\These_RBM\Laboratoire\Data_Base_Small\Data_Generees\CG"

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=DF)
Nb_simulations=DF.shape[0]
synthesizer_CG = GaussianCopulaSynthesizer(metadata)
synthesizer_CG.fit(DF)
AutoMPG_Generative = synthesizer_CG.sample(num_rows=Nb_simulations)
excel_path = os.path.join(folder_path, "AutoMPG_Generative.xlsx")
AutoMPG_Generative.to_excel(excel_path, index=False)

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=DF2)
Nb_simulations=DF2.shape[0]
synthesizer_CG = GaussianCopulaSynthesizer(metadata)
synthesizer_CG.fit(DF2)
Breast_Generative = synthesizer_CG.sample(num_rows=Nb_simulations)
excel_path = os.path.join(folder_path, "Breast_Generative.xlsx")
Breast_Generative.to_excel(excel_path, index=False)

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=DF3)
Nb_simulations=DF3.shape[0]
synthesizer_CG = GaussianCopulaSynthesizer(metadata)
synthesizer_CG.fit(DF3)
Cancer_Generative = synthesizer_CG.sample(num_rows=Nb_simulations)
excel_path = os.path.join(folder_path, "Cancer_Generative.xlsx")
Cancer_Generative.to_excel(excel_path, index=False)

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=DF4)
Nb_simulations=DF4.shape[0]
synthesizer_CG = GaussianCopulaSynthesizer(metadata)
synthesizer_CG.fit(DF4)
Ecoli_Generative = synthesizer_CG.sample(num_rows=Nb_simulations)
excel_path = os.path.join(folder_path, "Ecoli_Generative.xlsx")
Ecoli_Generative.to_excel(excel_path, index=False)

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=DF5)
Nb_simulations=DF5.shape[0]
synthesizer_CG = GaussianCopulaSynthesizer(metadata)
synthesizer_CG.fit(DF5)
FED_ECO_Generative = synthesizer_CG.sample(num_rows=Nb_simulations)
excel_path = os.path.join(folder_path, "FED_ECO_Generative.xlsx")
FED_ECO_Generative.to_excel(excel_path, index=False)

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=DF6)
Nb_simulations=DF6.shape[0]
synthesizer_CG = GaussianCopulaSynthesizer(metadata)
synthesizer_CG.fit(DF6)
Heart_Generative = synthesizer_CG.sample(num_rows=Nb_simulations)
excel_path = os.path.join(folder_path, "Heart_Generative.xlsx")
Heart_Generative.to_excel(excel_path, index=False)

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=DF7)
Nb_simulations=DF7.shape[0]
synthesizer_CG = GaussianCopulaSynthesizer(metadata)
synthesizer_CG.fit(DF7)
iris_Generative = synthesizer_CG.sample(num_rows=Nb_simulations)
excel_path = os.path.join(folder_path, "iris_Generative.xlsx")
iris_Generative.to_excel(excel_path, index=False)

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=DF8)
Nb_simulations=DF8.shape[0]
synthesizer_CG = GaussianCopulaSynthesizer(metadata)
synthesizer_CG.fit(DF8)
StatLog_Generative = synthesizer_CG.sample(num_rows=Nb_simulations)
excel_path = os.path.join(folder_path, "StatLog_Generative.xlsx")
StatLog_Generative.to_excel(excel_path, index=False)

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=DF8)
Nb_simulations=DF8.shape[0]
synthesizer_CG = GaussianCopulaSynthesizer(metadata)
synthesizer_CG.fit(DF8)
StatLog_Generative = synthesizer_CG.sample(num_rows=Nb_simulations)
excel_path = os.path.join(folder_path, "StatLog_Generative.xlsx")
StatLog_Generative.to_excel(excel_path, index=False)

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=DF9)
Nb_simulations=DF9.shape[0]
synthesizer_CG = GaussianCopulaSynthesizer(metadata)
synthesizer_CG.fit(DF9)
Student_Generative = synthesizer_CG.sample(num_rows=Nb_simulations)
excel_path = os.path.join(folder_path, "Student_Generative.xlsx")
Student_Generative.to_excel(excel_path, index=False)

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=DF10)
Nb_simulations=DF10.shape[0]
synthesizer_CG = GaussianCopulaSynthesizer(metadata)
synthesizer_CG.fit(DF10)
wine_Generative = synthesizer_CG.sample(num_rows=Nb_simulations)
excel_path = os.path.join(folder_path, "wine_Generative.xlsx")
wine_Generative.to_excel(excel_path, index=False)



###########################################
#Importation des bases de données générées#
###########################################

del AutoMPG_Generative,Breast_Generative,Cancer_Generative,Ecoli_Generative,FED_ECO_Generative,Heart_Generative,iris_Generative,StatLog_Generative,Student_Generative,wine_Generative,DF,DF2,DF3,DF4,DF5,DF6,DF7,DF8,DF9,DF10,dt_AutoMPG,dt_Breast,dt_Cancer,dt_Ecoli,dt_FED_ECO,dt_Heart,dt_StatLog,dt_Student,dt_genere,dt_iris,dt_wine

repertoire ="D:\Documents\Autre\These_RBM\Laboratoire\Data_Base_Small\Data_Generees\RBM"
dict_df = {}

for fichier in os.listdir(repertoire):
    if fichier.endswith('.xlsx'):
        chemin_fichier = os.path.join(repertoire, fichier)
        nom_df = fichier.split('.')[0]  
        dict_df[nom_df] = pd.read_excel(chemin_fichier, header=0)

AutoMPG_Generative=dict_df['AutoMPG_Generative']
Breast_Generative=dict_df['Breast_Generative']
Cancer_Generative=dict_df['Cancer_Generative']
Ecoli_Generative=dict_df['Ecoli_Generative']
FED_ECO_Generative=dict_df['FED_ECO_Generative']
Heart_Generative=dict_df['Heart_Generative']
iris_Generative=dict_df['iris_Generative']
StatLog_Generative=dict_df['StatLog_Generative']
Student_Generative=dict_df['Student_Generative']
wine_Generative=dict_df['wine_Generative']

#Itérer les datasets et récupérer les résultats sur la console
    

DF=wine_Generative
accuracy_list_RBM = []

for i in range(100):
    dt_genere = Machine_Boltzmann_Adaptative(DF)
        
    dt_1 = dt_genere[0]
    dt_2_RBM = dt_genere[1]
        
    dt_1['y'] = 1
    dt_2_RBM['y'] = 0
    
    result_RBM = pd.concat([dt_1, dt_2_RBM], ignore_index=True)
    train_set_RBM, test_set_RBM = train_test_split(result_RBM, test_size=0.3, random_state=1)
       
    categorical_cols = [col for col in result_RBM.columns if result_RBM[col].dtype == 'object']
    numerical_cols = [col for col in result_RBM.columns if col not in categorical_cols and col != 'y']

    preprocessor = ColumnTransformer(
         transformers=[
             ('num', StandardScaler(), numerical_cols), 
             ('cat', OneHotEncoder(), categorical_cols) 
        ])

    pipe = make_pipeline(preprocessor, LogisticRegression())
    pipe.fit(train_set_RBM.drop(columns=['y']), train_set_RBM['y'])
   
    y_test_pred_RBM = pipe.predict(test_set_RBM.drop(columns=['y']))

    accuracy_test_RBM = accuracy_score(test_set_RBM['y'], y_test_pred_RBM)
    print(f"Accuracy rate sur l'échantillon de test numéro {i} : {accuracy_test_RBM}")

    accuracy_list_RBM.append(accuracy_test_RBM)
     
accuracy_mean_RBM = np.mean(accuracy_list_RBM)
print(accuracy_mean_RBM)
accuracy_std_RBM = np.std(accuracy_list_RBM)
print(accuracy_std_RBM)
    


###########################################
#Importation des bases de données générées#
###########################################


repertoire ="D:\Documents\Autre\These_RBM\Laboratoire\Data_Base_Small\Data_Generees\CG"
dict_df = {}

for fichier in os.listdir(repertoire):
    if fichier.endswith('.xlsx'):
        chemin_fichier = os.path.join(repertoire, fichier)
        nom_df = fichier.split('.')[0]  
        dict_df[nom_df] = pd.read_excel(chemin_fichier, header=0)

AutoMPG_Generative=dict_df['AutoMPG_Generative']
Breast_Generative=dict_df['Breast_Generative']
Cancer_Generative=dict_df['Cancer_Generative']
Ecoli_Generative=dict_df['Ecoli_Generative']
FED_ECO_Generative=dict_df['FED_ECO_Generative']
Heart_Generative=dict_df['Heart_Generative']
iris_Generative=dict_df['iris_Generative']
StatLog_Generative=dict_df['StatLog_Generative']
Student_Generative=dict_df['Student_Generative']
wine_Generative=dict_df['wine_Generative']

#Itérer les datasets et récupérer les résultats sur la console
    
DF=wine_Generative
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=DF)
Nb_simulations=DF.shape[0]
synthesizer_CG = GaussianCopulaSynthesizer(metadata)
synthesizer_CG.fit(DF)
accuracy_list_CG = []


for i in range(100):    
    dt_1 = DF
    dt_1['y'] = 1
    
    dt_2_CG = synthesizer_CG.sample(num_rows=Nb_simulations)
    
    dt_2_CG['y'] = 0
    
    result_CG = pd.concat([dt_1, dt_2_CG], ignore_index=True)
    train_set_CG, test_set_CG = train_test_split(result_CG, test_size=0.3, random_state=1)
    
    
    categorical_cols = [col for col in result_CG.columns if result_CG[col].dtype == 'object']
    numerical_cols = [col for col in result_CG.columns if col not in categorical_cols and col != 'y']

    preprocessor = ColumnTransformer(
         transformers=[
             ('num', StandardScaler(), numerical_cols), 
             ('cat', OneHotEncoder(), categorical_cols)  
        ])

    pipe = make_pipeline(preprocessor, LogisticRegression())
    pipe.fit(train_set_CG.drop(columns=['y']), train_set_CG['y'])
   
    y_test_pred_CG = pipe.predict(test_set_CG.drop(columns=['y']))

    accuracy_test_CG = accuracy_score(test_set_CG['y'], y_test_pred_CG)
    print(f"Accuracy rate sur l'échantillon de test numéro {i} : {accuracy_test_CG}")

    accuracy_list_CG.append(accuracy_test_CG)
    
accuracy_mean_CG = np.mean(accuracy_list_CG)
print(accuracy_mean_CG)
accuracy_std_CG = np.std(accuracy_list_CG)
print(accuracy_std_CG)






