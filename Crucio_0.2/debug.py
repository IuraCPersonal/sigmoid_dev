from ADOMS import ADOMS
import pandas as pd


df = pd.read_csv('heart.csv')
test = ADOMS(binary_columns=['sex', 'target'])

test.balance(df, 'target')
