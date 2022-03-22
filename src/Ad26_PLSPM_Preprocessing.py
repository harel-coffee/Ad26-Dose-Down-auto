# Daniel Zhu
# May 21st, 2021
# Preprocessing in preparation for PLS-PM.
from Systems_Serology_Preprocessing import SystemsSerologyPreprocessing
import numpy as np
import pandas as pd


data = pd.read_csv('.\Processed_Data\SystemsSerology.csv', index_col=0)
# Invert the last two columns to get a measure that's positively correlated w/ the other features rather than
# negatively:
data.iloc[:,-2:] = data.iloc[:,-2:].apply(lambda x: 1/x, axis=1)
data.replace(np.inf, 1, inplace=True)
# Add a tiny amount of noise to the last two columns of data to prevent errors when bootstrapping.
data.iloc[:,-2:] = data.iloc[:,-2:] + np.random.normal(0, 1e-6, data.iloc[:,-2:].shape)
test = SystemsSerologyPreprocessing(data, ignore_index=True, output_basename="SystemsSerologyZscored")
test = test.z_score(save=True)