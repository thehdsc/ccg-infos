# Copyright 2020 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example code for the nodes in the example pipeline. This code is meant
just for illustrating basic Kedro features.

PLEASE DELETE THIS FILE ONCE YOU START WORKING ON YOUR OWN PROJECT!
"""

from typing import Any, Dict

import pandas as pd
from typing import Dict, List
from sklearn.preprocessing import LabelEncoder
from .synthesizer import generate_data_leadtime

def change_column_order(df, col_name, index):
    cols = df.columns.tolist()
    cols.remove(col_name)
    cols.insert(index, col_name)
    return df[cols]


def map_data(sndoc2: pd.DataFrame,
             sndoc22: pd.DataFrame) -> List:
    """Node for concat the different "sn_doc2", "MULTI_sn_mart", "MULTI_sn_fam", 
    "Sn_Doc2-dadosAdicionaisFio" and "MULTI_ct_terc1" and execute a set of transformations
    so that in the end of the process we can have a clean dataset with the following structure:
    "id130","id120", "mart_cod", "data_ocompra", "data_recebida", "terc1_cod", "doc2_qtdart", "data_prevista", "dias", "ldtime", "famil"
    """
    
    #Replace commas with dots and tranform quantities in floats
    sndoc2["doc2_qtdart"] = sndoc2["doc2_qtdart"].astype("str").str.replace(",",".").astype("float")
    sndoc2["doc2_qtdtrat"] = sndoc2["doc2_qtdtrat"].astype("str").str.replace(",",".").astype("float")
    sndoc22["doc2_qtdart"] = sndoc22["doc2_qtdart"].astype("str").str.replace(",",".").astype("float")
    sndoc22["doc2_qtdtrat"] = sndoc22["doc2_qtdtrat"].astype("str").str.replace(",",".").astype("float")

    #Select documents with 120 as origin and 130 as reception
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
    
    #1Map DataFrame
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

    return [orig120, orig122, orig124, orig130, orig132, orig134 ]

def concat_data(orig120: pd.DataFrame,
                orig122: pd.DataFrame,
                orig124: pd.DataFrame,
                orig130: pd.DataFrame,
                orig132: pd.DataFrame,
                orig134: pd.DataFrame) -> pd.DataFrame:
    """Node for merge the previous 'orig120', 'orig122', 'orig124', 'orig130', 'orig132', 'orig134'
    and concat them. At the same time some known misplaced dates are corrected.  """
    
    fin1 = pd.merge(orig130, orig120, on='id120', how='inner')
    fin2 = pd.merge(orig132, orig122, on='id120', how='inner')
    fin3 = pd.merge(orig134, orig124, on='id124', how='inner')

    fin3.rename(columns={'id134': 'id130'}, inplace=True)
    fin3.rename(columns={'id124': 'id120'}, inplace=True)

    #2Concat DataFrames
    final = pd.concat([fin1, fin2, fin3], axis=0)

    #2Drop 'doc2_qtdreceb'
    final.drop(['doc2_qtdreceb'], axis=1, inplace=True)

    #2Reorder columns
    final = change_column_order(final, 'data_ocompra',3)
    final = change_column_order(final, 'data_recebida',4)

    #2Transformations on errors
    final["data_prevista"] = final["data_prevista"].replace("27/06/9201","27/06/2019")
    final["data_prevista"] = final["data_prevista"].replace("19/04/0218","19/04/2018")

    intermediate_01 = final

    return intermediate_01

def select_data(intermediate_01: pd.DataFrame,
                snmart: pd.DataFrame,
                snfam: pd.DataFrame,
                forn: pd.DataFrame) -> pd.DataFrame:
    
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


def split_data(primary_01: pd.DataFrame) -> List:
    """Node for split previous contact DataFrame and create two new
    DataFrames por different purposes: "leadtime" and "days"
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

def generate_synthetic_data(feature_leadtime: pd.DataFrame, parameters: Dict) -> List:

    synthetic_leadtime_01 = generate_data_leadtime(feature_leadtime, parameters["synthesizer_01"])
    synthetic_leadtime_02 = generate_data_leadtime(feature_leadtime, parameters["synthesizer_02"])

    return [synthetic_leadtime_01, synthetic_leadtime_02]


