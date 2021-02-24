from typing import Any, Dict

import pandas as pd
import numpy as np
from typing import Dict, List
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from .synthesizer import generate_data_leadtime, generate_data_days

def change_column_order(df, col_name, index):
    """Auxiliar function to change column order

        Args:
            df: Data.
            col_name: Name of the column to be reordered.
            index: Index where the user wants the column to go.
        Returns:
            A dataframe reordered.

    """
    cols = df.columns.tolist()
    cols.remove(col_name)
    cols.insert(index, col_name)
    return df[cols]


def origin_preprocessing(sndoc2: pd.DataFrame,
                         sndoc22: pd.DataFrame) -> List:
    """Replaces the decimal character from comma to point and transforms quantities to float type. Creates six different datasets 
    based on the source and receipt codes of the documents and renames a set of fields for a better understanding.

        Args:
            sndoc2: Documents data.
            sndoc22: Aditional documents data.
        Returns:
            A List with six pandas DataFrame containing the documents from different origins.

    """
    #Replace comma with point and tranform quantities in floats
    sndoc2["doc2_qtdart"] = sndoc2["doc2_qtdart"].astype("str").str.replace(",",".").astype("float")
    sndoc2["doc2_qtdtrat"] = sndoc2["doc2_qtdtrat"].astype("str").str.replace(",",".").astype("float")
    sndoc22["doc2_qtdart"] = sndoc22["doc2_qtdart"].astype("str").str.replace(",",".").astype("float")
    sndoc22["doc2_qtdtrat"] = sndoc22["doc2_qtdtrat"].astype("str").str.replace(",",".").astype("float")

    orig120 = sndoc2.loc[(sndoc2["cdoc_cod"] == 120),["empr_cod", "cdoc_cod", "doc1_num", "mart_cod", "doc2_linha","doc2_qtdart",
                                                  "doc2_qtdtrat","doc2_pedida","doc1_emiss", "terc1_cod"]]

    orig130 = sndoc2.loc[(sndoc2["cdoc_cod"] == 130) & (sndoc2["cdoc_origcod"] == 120),["empr_cod", "cdoc_cod", "doc1_num", 
                                                                                    "mart_cod","doc2_linha","doc2_origlinha","cdoc_origcod",
                                                                                    "doc1_orignum","doc1_emiss", "terc1_cod"]]
    orig124 = sndoc2.loc[(sndoc2["cdoc_cod"] == 124),["empr_cod", "cdoc_cod", "doc1_num", "mart_cod", "doc2_linha","doc2_qtdart",
                                                  "doc2_qtdtrat","doc2_pedida","doc1_emiss", "terc1_cod"]]

    orig134 = sndoc2.loc[(sndoc2["cdoc_cod"] == 134) & (sndoc2["cdoc_origcod"] == 124),["empr_cod", "cdoc_cod", "doc1_num", 
                                                                                    "mart_cod","doc2_linha","doc2_origlinha","cdoc_origcod",
                                                                                    "doc1_orignum","doc1_emiss", "terc1_cod"]]

    orig122 = sndoc22.loc[(sndoc22["cdoc_cod"] == 120),["empr_cod", "cdoc_cod", "doc1_num", "mart_cod", "doc2_linha","doc2_qtdart",
                                                  "doc2_qtdtrat","doc2_pedida","doc1_emiss", "terc1_cod"]]

    orig132 = sndoc22.loc[(sndoc22["cdoc_cod"] == 130) & (sndoc22["cdoc_origcod"] == 120),["empr_cod", "cdoc_cod", "doc1_num", 
                                                                                    "mart_cod","doc2_linha","doc2_origlinha","cdoc_origcod",
                                                                                    "doc1_orignum","doc1_emiss", "terc1_cod"]]

    #Rename columns for a better understanding
    orig120.rename(columns={'doc2_qtdtrat': 'doc2_qtdreceb'}, inplace=True)
    orig120.rename(columns={'doc1_emiss': 'data_ocompra'}, inplace=True)
    orig120.rename(columns={'doc2_pedida': 'data_prevista'}, inplace=True)
    orig130.rename(columns={'doc1_emiss': 'data_recebida'}, inplace=True)
    orig124.rename(columns={'doc2_qtdtrat': 'doc2_qtdreceb'}, inplace=True)
    orig124.rename(columns={'doc1_emiss': 'data_ocompra'}, inplace=True)
    orig124.rename(columns={'doc2_pedida': 'data_prevista'}, inplace=True)
    orig134.rename(columns={'doc1_emiss': 'data_recebida'}, inplace=True)
    orig122.rename(columns={'doc2_qtdtrat': 'doc2_qtdreceb'}, inplace=True)
    orig122.rename(columns={'doc1_emiss': 'data_ocompra'}, inplace=True)
    orig122.rename(columns={'doc2_pedida': 'data_prevista'}, inplace=True)
    orig132.rename(columns={'doc1_emiss': 'data_recebida'}, inplace=True)
    
    #Map DataFrame
    orig120['id120'] = orig120['empr_cod'].map(str) + orig120['cdoc_cod'].map(str) + orig120['doc1_num'].map(str) + orig120['doc2_linha'].map(str)
    orig120.drop(['empr_cod', 'cdoc_cod','mart_cod','doc1_num','doc2_linha'], axis=1, inplace=True)
    orig130['id130'] = orig130['empr_cod'].map(str) + orig130['cdoc_cod'].map(str) + orig130['doc1_num'].map(str) + orig130['doc2_linha'].map(str)
    orig130['id120'] = orig130['empr_cod'].map(str) + orig130['cdoc_origcod'].map(str) + orig130['doc1_orignum'].map(str) + orig130['doc2_origlinha'].map(str)
    orig130.drop(['empr_cod', 'cdoc_cod','doc1_num','doc2_linha','cdoc_origcod','doc1_orignum','doc2_origlinha' ], axis=1, inplace=True)
    orig124['id124'] = orig124['empr_cod'].map(str) + orig124['cdoc_cod'].map(str) + orig124['doc1_num'].map(str) + orig124['doc2_linha'].map(str)
    orig124.drop(['empr_cod', 'cdoc_cod','mart_cod','doc1_num','doc2_linha'], axis=1, inplace=True)
    orig134['id134'] = orig134['empr_cod'].map(str) + orig134['cdoc_cod'].map(str) + orig134['doc1_num'].map(str) + orig134['doc2_linha'].map(str)
    orig134['id124'] = orig134['empr_cod'].map(str) + orig134['cdoc_origcod'].map(str) + orig134['doc1_orignum'].map(str) + orig134['doc2_origlinha'].map(str)
    orig134.drop(['empr_cod', 'cdoc_cod','doc1_num','doc2_linha','cdoc_origcod','doc1_orignum','doc2_origlinha' ], axis=1, inplace=True)
    orig122['id120'] = orig122['empr_cod'].map(str) + orig122['cdoc_cod'].map(str) + orig122['doc1_num'].map(str) + orig122['doc2_linha'].map(str)
    orig122.drop(['empr_cod', 'cdoc_cod','mart_cod','doc1_num','doc2_linha'], axis=1, inplace=True)
    orig132['id130'] = orig132['empr_cod'].map(str) + orig132['cdoc_cod'].map(str) + orig132['doc1_num'].map(str) + orig132['doc2_linha'].map(str)
    orig132['id120'] = orig132['empr_cod'].map(str) + orig132['cdoc_origcod'].map(str) + orig132['doc1_orignum'].map(str) + orig132['doc2_origlinha'].map(str)
    orig132.drop(['empr_cod', 'cdoc_cod','doc1_num','doc2_linha','cdoc_origcod','doc1_orignum','doc2_origlinha' ], axis=1, inplace=True)

    #Order columns 
    orig120 = change_column_order(orig120, 'id120', 0)
    orig130 = change_column_order(orig130, 'id130', 0)
    orig130 = change_column_order(orig130, 'id120', 1)
    orig124 = change_column_order(orig124, 'id124', 0)
    orig134 = change_column_order(orig134, 'id134', 0)
    orig134 = change_column_order(orig134, 'id124', 1)
    orig122 = change_column_order(orig122, 'id120', 0)
    orig132 = change_column_order(orig132, 'id130', 0)
    orig132 = change_column_order(orig132, 'id120', 1)

    return [orig120, orig122, orig124, orig130, orig132, orig134]

def concat_data(orig120: pd.DataFrame,
                orig122: pd.DataFrame,
                orig124: pd.DataFrame,
                orig130: pd.DataFrame,
                orig132: pd.DataFrame,
                orig134: pd.DataFrame) -> pd.DataFrame:
    """Merges the previous 'orig120', 'orig122', 'orig124', 'orig130', 'orig132', 'orig134'
    created Dataframes and concat them. At the same time some known misplaced dates are corrected.

        Args:
            orig120: Documents with origin 120.
            orig122: Documents with origin 122.
            orig124: Documents with origin 124.
            orig130: Documents with origin 130.
            orig132: Documents with origin 132.
            orig134: Documents with origin 134.
        Returns:
            A pandas DataFrame 'intermediate_01' containing all the documents merged and concatenated.

    """
    fin1 = pd.merge(orig130, orig120, on='id120', how='inner')
    fin2 = pd.merge(orig132, orig122, on='id120', how='inner')
    fin3 = pd.merge(orig134, orig124, on='id124', how='inner')

    fin3.rename(columns={'id134': 'id130'}, inplace=True)
    fin3.rename(columns={'id124': 'id120'}, inplace=True)

    #Concat DataFrames
    final = pd.concat([fin1, fin2, fin3], axis=0)

    #Drop 'doc2_qtdreceb'
    final.drop(['doc2_qtdreceb'], axis=1, inplace=True)

    #Reorder columns
    final = change_column_order(final, 'data_ocompra',3)
    final = change_column_order(final, 'data_recebida',4)

    #Transformations on errors
    final["data_prevista"] = final["data_prevista"].replace("27/06/9201","27/06/2019")
    final["data_prevista"] = final["data_prevista"].replace("19/04/0218","19/04/2018")

    intermediate_01 = final

    return intermediate_01

def select_data(intermediate_01: pd.DataFrame,
                snmart: pd.DataFrame,
                snfam: pd.DataFrame,
                forn: pd.DataFrame) -> pd.DataFrame:
    """Calculates leadtime and days, and map the intermediate dataset with family, product and supllier.

        Args:
            intermediate_01: Documents transformed data.
            snmart: Data with products.
            snfam: Data with family.
            forn: Data with suppliers.
        Returns:
            A pandas DataFrame 'primary_01' containing all the documents mapped with all their information.

    """
    fornecedor = forn.loc[(forn["empr_cod"] == 1),["terc1_cod", "pais_cod"]]
    intermediate_01[['data_recebida','data_prevista', 'data_ocompra']] = intermediate_01[['data_recebida','data_prevista', 'data_ocompra']].apply(pd.to_datetime, format= "%d/%m/%Y")
    intermediate_01['dias'] = (intermediate_01['data_recebida'] - intermediate_01['data_prevista']).dt.days

    intermediate_01['ldtime'] = (intermediate_01['data_recebida'] - intermediate_01['data_ocompra']).dt.days

    intermediate_01['bool'] = (intermediate_01['terc1_cod_x'] == intermediate_01['terc1_cod_y'])
    intermediate_01.drop(['bool', 'terc1_cod_y'], axis=1, inplace=True)
    intermediate_01.rename(columns={'terc1_cod_x': 'terc1_cod'}, inplace=True)

    #Lista de produtos e respetiva familia
    produto = snmart.loc[(snmart["empr_cod"] == 1),["fam_cod","mart_cod", "mart_descr1"]]
    familia = snfam.loc[(snfam["empr_cod"] == 1),["fam_cod","fam_descr1"]]

    #Criar dicionario de familias
    fam_dict = dict(zip(familia['fam_cod'], familia['fam_descr1']))

    #substituir na dataset de dos produtos
    produto['fam_cod']= produto['fam_cod'].map(fam_dict).fillna(produto['fam_cod'])

    #dicionario para os produtos por familia
    prod_dict = dict(zip(produto['mart_cod'], produto['fam_cod']))

    #dicionario para o fornecedor
    for_dict = dict(zip(fornecedor['terc1_cod'], fornecedor['pais_cod']))
    intermediate_01['terc1_cod']= intermediate_01['terc1_cod'].map(for_dict).fillna(intermediate_01['terc1_cod'])

    #Criar coluna nova no dataset final
    intermediate_01["famil"] = intermediate_01['mart_cod'].map(prod_dict)

    #remover linhas de outros tipos de produtos
    options = ['ACESSÓRIOS DE EMBALAGEM', 'ACESSÓRIOS GERAIS', 'SERVIÇOS','DIVERSOS', 'PRODUTO ACABADO'] 
    intermediate_01 = intermediate_01[-intermediate_01["famil"].isin(options)]

    #Drop misplaced entry
    intermediate_01.drop(intermediate_01[intermediate_01.doc2_qtdart == 0.001].index, inplace=True)

    primary_01 = intermediate_01

    return primary_01


def split_data_targets(primary_01: pd.DataFrame) -> List:
    """Preprocessing of leadtime and days adding boolean columns based if 'data_recebida' is equal or greater that 'data_ocompra' and
    if 'data_prevista' is not NaT, create two different DataFrame based on target, drop rows based on previous preprocessing 
    drop duplicates and label encode (sklearn.preprocessing.LabelEncoder) 'mart_cod'.

        Args:
            primary_01: Merged, concatenated and preprocessed raw data.
        Returns:
            A List with two pandas DataFrame for the different 'leadtime' and 'days' targets.

    """

    primary_01["bool_leadtime"] = (primary_01["data_recebida"] >= primary_01["data_ocompra"])
    if (primary_01["data_prevista"] is pd.NaT):
        primary_01["bool_days"] = False
    else:
        primary_01["bool_days"] = (primary_01["data_prevista"] >= primary_01["data_ocompra"])

    #Criar dataset para leadtime e para os dias 
    leadtime = primary_01.loc[:,['terc1_cod', 'mart_cod', 'doc2_qtdart', 'data_ocompra', 'data_recebida', 'ldtime', 'bool_leadtime']]
    days = primary_01.loc[:,['terc1_cod', 'mart_cod', 'doc2_qtdart','data_ocompra', 'data_prevista', 'data_recebida','dias', 'bool_days']]

    #Remover linhas onde o bool_leadtime é falso e, posteriormente, remover a coluna bool_leadtime
    leadtime.drop(leadtime[leadtime.bool_leadtime == False].index, inplace=True)
    leadtime.drop(['bool_leadtime'], axis=1, inplace=True)

    #Drop 'terc1_cod'
    leadtime.drop(['terc1_cod'], axis=1, inplace=True)

    #Check duplicates
    duplicateRowsDF = leadtime[leadtime.duplicated(['mart_cod', 'doc2_qtdart','data_ocompra'])]

    grouped_multiple = leadtime.groupby(['mart_cod', 'doc2_qtdart','data_ocompra']).agg({'ldtime': ['mean']})
    grouped_multiple.columns = ['ldtime_mean']
    grouped_multiple = grouped_multiple.reset_index()

    leadtime = grouped_multiple

    leadtime['ldtime_mean'] = leadtime['ldtime_mean'].round(0)
    leadtime.rename(columns={'ldtime_mean': 'ldtime'}, inplace=True)

    leadtime.drop(leadtime[leadtime.ldtime >= 180].index, inplace=True)

    #Label encoder
    leadtime['mart_cod'] = LabelEncoder().fit_transform(leadtime['mart_cod'])

    ##DAYS
    days.drop(days[days.bool_days == False].index, inplace=True)
    days.drop(['bool_days'], axis=1, inplace=True)

    #Drop 'terc1_cod'
    days.drop(['terc1_cod'], axis=1, inplace=True)

    duplicateRows = days[days.duplicated(['mart_cod', 'doc2_qtdart','data_prevista' ])]

    grouped = days.groupby(['mart_cod', 'doc2_qtdart', 'data_prevista' ]).agg({'dias': ['mean']})
    grouped.columns = ['dias_mean']
    grouped= grouped.reset_index()

    days=grouped

    days['dias_mean'] = days['dias_mean'].round(0)
    days.rename(columns={'dias_mean': 'dias'}, inplace=True)

    #Eliminar observações em que os dias sejam iguais ou inferiores a -60 e iguais ou superiores a 180
    days.drop(days[days.dias <= -60].index, inplace=True)
    days.drop(days[days.dias >= 180].index, inplace=True)

   #Label Encoder 
    days['mart_cod'] = LabelEncoder().fit_transform(days['mart_cod'])
    
    feature_leadtime = leadtime
    feature_days = days

    return [feature_leadtime, feature_days]

def generate_synthetic_data_leadtime(feature_leadtime: pd.DataFrame) -> List:
    """Generate synthetic data using Synthetic Data Vault (sdv.dev) CTGAN and GaussianCopula
    methods for leadtime.

        Args:
            feature_leadtime: Data with the leadtime target.
        Returns:
            A List with two pandas DataFrame containing synthetic data generated by CTGAN and the other using
            GaussianCopula.
    """

    synthetic_leadtime_01, synthetic_leadtime_02 = generate_data_leadtime(feature_leadtime)

    return [synthetic_leadtime_01, synthetic_leadtime_02]

def generate_synthetic_data_days(feature_days: pd.DataFrame) -> List:
    """Generate synthetic data using Synthetic Data Vault (sdv.dev) CTGAN and GaussianCopula
    methods for days.

        Args:
            feature_days: Data with the days target.
        Returns:
            A List with two pandas DataFrame containing synthetic data generated by CTGAN and the other using
            GaussianCopula.

    """
    synthetic_days_01, synthetic_days_02 = generate_data_days(feature_days)

    return [synthetic_days_01, synthetic_days_02]


def mix_real_synthtetic_leadtime(real_data: pd.DataFrame,
                                synthethic_data: pd.DataFrame,
                                parameters: Dict) -> pd.DataFrame:
    """Concat real and synthetic leadtime data, shuffle, convert date fields to unix timestamp, standardize all fields except
    the target 'leadtime' and drop rows based on a given target min and max.

        Args:
            real_data: Real leadtime data.
            synthethic_data: Synthetic leadtime data.
            parameters: Parameters defined in parameters.yml
        Returns:
            A pandas DataFrame with the leadtime data output.
    """

    frames = [real_data, synthethic_data]
    #Concat the real and synthetic DataFrames
    result = pd.concat(frames)
    #Reset concat output index
    result = result.sample(frac=1).reset_index(drop=True)
    #APPLY UNIX TIMESTAMP TO LEADTIME DATETIME COLUMN 'data_ocompra'
    result['data_ocompra'] = pd.to_datetime(result['data_ocompra']).astype(np.int64)
    #STANDARDIZE LEADTIME DATAFRAME
    result[['mart_cod','doc2_qtdart','data_ocompra']] = StandardScaler().fit_transform(result[['mart_cod','doc2_qtdart','data_ocompra']])
    #Apply min to leadtime
    result.drop(result[result.ldtime < parameters['leadtime']['min']].index, inplace=True)
    #Apply max to leadtime
    result.drop(result[result.ldtime > parameters['leadtime']['max']].index, inplace=True)
    return result

def mix_real_synthtetic_days(real_data: pd.DataFrame,
                            synthethic_data: pd.DataFrame,
                            parameters: Dict) -> pd.DataFrame:
    """Concat real and synthetic days data, shuffle, convert date fields to unix timestamp, standardize all fields except
    the target 'dias' and drop rows based on a given target min and max.

        Args:
            real_data: Real days data.
            synthethic_data: Synthetic days data.
            parameters: Parameters defined in parameters.yml
        Returns:
            A pandas DataFrame with the Days data output.
    """
    frames = [real_data, synthethic_data]
    #Concat the real and synthetic DataFrames
    result = pd.concat(frames)
    #Shuffle and Reset concat output index
    result = result.sample(frac=1).reset_index(drop=True)
    #APPLY UNIX TIMESTAMP TO LEADTIME DATETIME COLUMN 'data_ocompra'
    result['data_prevista'] = pd.to_datetime(result['data_prevista']).astype(np.int64)
    #STANDARDIZE LEADTIME DATAFRAME
    result[['mart_cod','doc2_qtdart','data_prevista']] = StandardScaler().fit_transform(result[['mart_cod','doc2_qtdart','data_prevista']])
    #Apply min to leadtime
    result.drop(result[result.dias < parameters['days']['min']].index, inplace=True)
    #Apply max to leadtime
    result.drop(result[result.dias > parameters['days']['max']].index, inplace=True)
    return result












