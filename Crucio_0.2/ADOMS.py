'''
Created with love by Sigmoid

@Author - Cius Iurie - iurie.cius.personal@gmail.com
'''

import sys
import random
import pandas as pd
import numpy as np
from random import randrange
from sklearn.decomposition import PCA

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


class DifferentColumnsError(BaseException):
    pass


class ADOMS:

    def __init__(self, k: "int > 0" = 5, seed: float = 42, binary_columns: list = None, n_clusters: list = None) -> None:
        '''
            Setting up the algorithm
        :param k: int, k>0, default = 5
            Number of neighbours which will be considered when looking for simmilar data points
        :param seed: intt, default = 42
            seed for random
        :param binary_columns: list, default = None
            The list of columns that should have binary values after balancing.
        '''

        self.__k = k
        self.seed = seed
        self.binary_columns = binary_columns
        self.n_clusters = n_clusters

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
            threshold = (self.df[column_name].max() +
                         self.df[column_name].min()) / 2
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

        # Check if the passed target argument is a column presented in the passed Data Frame.
        if target not in df.columns:
            raise NoSuchColumn(f"{target} isn't a column of passed data frame")

        # Check if all values passed in the binary_columns in the constructor are presented in the passed Data Frame.
        # mask = [column in df.columns for column in self.binary_columns]
        # if any(mask) != '--':
        #     raise DifferentColumnsError(
        #         f"The passed data frame doesn't contain the {np.ma.masked_array(self.binary_columns, mask)} columns passed to the binary columns.")

        # Check for binary.
        unique = df[target].unique()

        if len(unique)!=2:
            raise NotBinaryData(f"{target} column isn't a binary column")

        if target not in df.columns:
            raise NoSuchColumn(f"{target} isn't a column of passed data frame")

        # Creating an internal copy of the data frame.
        self.df = df.copy()
        self.target = target
        self.X_columns = [column for column in self.df.columns if column != target]

        # 1 - Randomly select one of the minority class examples pi in the original training data as the processing example.
        
        #check the minority class
        first_class = len(df[df[target]==unique[0]])/len(df[target])
        if first_class > 0.5:
            self.minority_class, self.majority_class = unique[1], unique[0]
        else: 
            self.minority_class, self.majority_class = unique[0], unique[1]
        
        self.majority = df[df[target]==self.majority_class]
        self.minority = df[df[target]==self.minority_class]

        self.minority_samples = self.df[self.df[target] == self.minority_class][self.X_columns].values
        self.majority_samples = self.df[self.df[target] == self.majority_class][self.X_columns].values

        #random example from minority class
        index = randrange(len(self.minority_samples))
        example = self.minority_samples[index]
        print(example)

        #select k neighbouors from this example
        neighbours_indexes = self.__get_k_neighbours(example) 

        print("KNN")
        print(neighbours_indexes)

        data_distribution = self.df.iloc[neighbours_indexes]

        pca = PCA(n_components = len(neighbours_indexes))
        pca.fit(data_distribution)
        # print(pca.components_[0])
        
        random_neighbour = self.minority_samples[random.choice(neighbours_indexes)]
        print(random_neighbour)

        #compute euclidian distance
        distance = np.linalg.norm(example - random_neighbour)
        print(distance)


    
    def __populate(self, N, i, nnaray, min_samples, k):
        while N:
            nn = random.randint(0, k - 2)

            diff = min_samples[nnaray[nn]] - min_samples[i]
            gap = random.uniform(0, 1)

            self.synthetic_arr[self.newindex, :] = min_samples[i] + gap * diff
            self.newindex += 1

            N -= 1


    def __get_k_neighbours(self,example):
        '''
            KNN, getting nearest neighbors
        :param example: Numpy array
            the sample row from minority class to get neighbours from
        :param minority_samples: Numpy.ndarray
            minority class samples from where we find neighbours
        '''
        distances = []
        
        for x in self.minority_samples:
            distances.append(np.linalg.norm(x - example, ord=2))
        predicted_index = np.argsort(distances)[1:self.__k + 1]

        return predicted_index
