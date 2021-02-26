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

Delete this when you start working on your own Kedro project.
"""
# pylint: disable=invalid-name

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd 
from sklearn.model_selection import KFold
from h2o.automl import H2OAutoML
import h2o
from sklearn.metrics import mean_absolute_error
import os


def create_kfolds(data: pd.DataFrame, parameter): 
    #Create paths
    paths = [
    'data/09_KFolds/'
    ]

    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path) 
        kf = KFold(n_splits=10, random_state=42, shuffle=True)
        kf.split(data)
    
        indexes = []

        for x in kf.split(data):
            indexes.append(list(x[1]))

        folds = {
            "1": None,
            "2": None,
            "3": None,
            "4": None,
            "5": None,
            "6": None,
            "7": None,
            "8": None,
            "9": None,
            "10": None,
        }

        for x in range(1,11):
            folds[str(x)] = data[data.index.isin(indexes[x-1])]

        folds["1"].to_csv(CTGAN_leadtime + "-fold1.csv", index=False)
        folds["2"].to_csv(CTGAN_leadtime + "-fold2.csv", index=False)
        folds["3"].to_csv(CTGAN_leadtime + "-fold3.csv", index=False)
        folds["4"].to_csv(CTGAN_leadtime + "-fold4.csv", index=False)
        folds["5"].to_csv(CTGAN_leadtime + "-fold5.csv", index=False)
        folds["6"].to_csv(CTGAN_leadtime + "-fold6.csv", index=False)
        folds["7"].to_csv(CTGAN_leadtime + "-fold7.csv", index=False)
        folds["8"].to_csv(CTGAN_leadtime + "-fold8.csv", index=False)
        folds["9"].to_csv(CTGAN_leadtime + "-fold9.csv", index=False)
        folds["10"].to_csv(CTGAN_leadtime + "-fold10.csv", index=False)

