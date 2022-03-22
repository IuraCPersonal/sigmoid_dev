'''
Created with love by Sigmoid

@Author - Cius Iurie - iurie.cius.personal@gmail.com
'''

import sys
import pandas as pd
import numpy as np
from crucio import SMOTE

class NotBetweenZeroAndOneError(BaseException):
    ''' Raised when a value is not between zero and one, but it should be'''
    pass

class NotBinaryData(BaseException):
    ''' Raised when the data passed is not binary '''
    pass

class NoSuchMethodError(BaseException):
    ''' Raised when the balancer or reducer doesnt't have a method '''
    pass

class NoSuchColumn(BaseException):
    ''' Raised when the data frame passed doesn't have a certain column '''
    pass

class SMOTERSB:

    def __init__(self, k: "int > 0" = 5, seed: float = 42, binary_columns: list = None) -> None:
        '''
            Setting up the algorithm
        :param k: int, k>0, default = 5
            Number of neighbours which will be considered when looking for simmilar data points
        :param seed: intt, default = 42
            seed for random
        :param binary_columns: list, default = None
            The list of columns that should have binary values after balancing.
        '''

        self.k = k
        self.seed = seed
        self.binary_columns = binary_columns

    def __infinity_check(self, matrix: 'np.array') -> 'np.array':
        '''
            This function replaces the infinity and -infinity values with the minimal and maximal float python values.
        :param matrix: 'np.array'
            The numpy array that was generated my the algorithm.
        :return: 'np.array'
            The numpy array with the infinity replaced values.
        '''

        matrix[matrix == -np.inf] = sys.float_info.min
        matrix[matrix == np.inf] = sys.float_info.max
        return matrix
    
    def __to_binary(self) -> None:
        '''
            If the :param binary_columns: is set to True then the intermediate values in binary columns will be rounded.
        '''

        for column_name in self.__binary_columns:
            serie = self.synthetic_df[column_name].values
            threshold = (self.df[column_name].max() + self.df[column_name].min()) / 2
            for i in range(len(serie)):
                if serie[i] >= threshold:
                    serie[i] = self.df[column_name].max()
                else:
                    serie[i] = self.df[column_name].min()
            self.synthetic_df[column_name] = serie

    def balance(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        '''
            Reducing the dimensionality of the data
        :param df: pandas DataFrame
             Data Frame on which the algorithm is applied
        :param target: string
             The target name of the value that we have to predict
        '''

        #check for binary
        unique = df[target].unique()
        if len(unique)!=2:
            raise NotBinaryData(f"{target} column isn't a binary column")

        if target not in df.columns:
            raise NoSuchColumn(f"{target} isn't a column of passed data frame")

        self.target= target
        self.df = df.copy()
        self.X_columns = [column for column in self.df.columns if column != target]
        