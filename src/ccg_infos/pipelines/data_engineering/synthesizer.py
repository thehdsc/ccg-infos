import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sdv.evaluation import evaluate
from sdv.constraints import Constraint
from sdv.tabular import CTGAN
from sdv.tabular import GaussianCopula

from sdv.constraints import Constraint

class  MinMaxQuantity(Constraint):
    """Ensure that doc2_qtdart follows a minimum and maximum"""

    def __init__(self, quantidade, min):
        self._quantidade = quantidade
        self._min = min

    def is_valid(self, table_data):
        """Say if quantity is equal or greater than given min"""
        egt = table_data[self._quantidade] >= self._min
        return egt

    
class  CodigoArtigo(Constraint):
    """Ensure that mart_cod follows a given minimum and maximum"""

    def __init__(self, codigo_artigo, min):
        self._codigoartigo = codigo_artigo
        self._min = min

    def is_valid(self, table_data):
        """Say if mart_cod is equal or greater than given minium."""
        egt = table_data[self._codigoartigo] >= self._min
        return egt

       
class  PositiveLeadtime(Constraint):
    """Ensure that leadtime follows a given minimum"""

    def __init__(self, leadtime, min):
        self._leadtime = leadtime
        self._min = min

    def is_valid(self, table_data):
        """Say if leadtime is equal or greater than given minimum."""
        egt = table_data[self._leadtime] >= self._min
        return egt

#Ensure that doc2_qtdart follows the minimum and maximum
quantidade = MinMaxQuantity(
        quantidade='doc2_qtdart',
        min=0
)

artigo = CodigoArtigo(
        codigo_artigo='mart_cod',
        min=0
)

leadtime = PositiveLeadtime(
        leadtime='ldtime',
        min=0
)

#store all the constraints that we will use in our model
constraints_leadtime = [
    quantidade,
    artigo,
    leadtime
    ]


def generate_data_leadtime(feature_leadtime, synthesizer):
    feature_leadtime['data_ocompra'] = pd.to_datetime(feature_leadtime['data_ocompra'])
    model = CTGAN(constraints=constraints_leadtime, field_transformers={
                                'doc2_qtdart': 'integer',
                                'data_ocompra': 'datetime',
                                'ldtime': 'integer'
                            })
    model.fit(feature_leadtime)
    synthetic_leadtime_data = model.sample(len(feature_leadtime))
    return synthetic_leadtime_data