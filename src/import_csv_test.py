# import time
# import pandas as pd

# # power_list = pd.read_csv("database\\decision_power_8_0927-115200.csv", skiprows=[1,3])
# power_list = pd.read_csv("database\\decision_power_8_0927-115200.csv")
# # power_list = pd.read_csv("database\\decision_power_40_1019-184345.csv")
# print(power_list.info())

# print(power_list.loc[0:1].values.tolist())

import numpy as np
halfway_number = np.array([0.125])
halfway_number[0]

print(halfway_number.astype('float64')[0])
print(halfway_number.astype('float32')[0])
print(halfway_number.astype('float16')[0])