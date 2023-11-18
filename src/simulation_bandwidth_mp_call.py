import simulation_bandwidth_mp as sb
import multiprocessing as mp
import time
import datetime
import os
import csv
import pandas as pd
import numpy as np

# CORE          = 4
# START_PROGRAM = 0
# # END_PROGRAM   = 3930719

# num = 180 #km
# output_csv = True
# result_process_num = list()
# power_list_process = list()

# now         = datetime.datetime.now()
# output_time = now.strftime("%m%d-%H%M%S")
# output_path = "Result\\Result_" + output_time

# if __name__ == "__main__":

#     power_list = pd.read_csv("database\\decision_power_32_1102-122930.csv", dtype=np.float32)
#     # power_list = pd.read_csv("database\\decision_power_8_1102-003616.csv", dtype=np.float16)

#     END_PROGRAM = len(power_list)

#     if output_csv:
#         os.mkdir(output_path)

#     start_end_list = list()

#     process_width = (END_PROGRAM - START_PROGRAM) / CORE

#     for process_num in range(CORE) :
#         start_process_num = START_PROGRAM + process_num * process_width 
#         end_process_num   = start_process_num + process_width
#         # start_end_list.append([power_list, process_num, START_PROGRAM, process_width, output_time, output_path])
#         start_end_list.append([power_list, process_num, int(start_process_num), int(end_process_num), output_time, output_path])
#     p = mp.Pool(CORE)
#     p.map(sb.main, start_end_list)


CORE          = 4
START_PROGRAM = [1251006, 5887200, 9714016, 13351587]
END_PROGRAM   = [3845234, 7690468, 11535702, 15380937]

num = 180 #km
output_csv = True
result_process_num = list()
power_list_process = list()

now         = datetime.datetime.now()
output_time = now.strftime("%m%d-%H%M%S")
output_path = "Result\\Result_" + output_time

if __name__ == "__main__":

    power_list = pd.read_csv("database\\decision_power_32_1102-122930.csv", dtype=np.float32)
    # power_list = pd.read_csv("database\\decision_power_8_1102-003616.csv", dtype=np.float16)

    if output_csv:
        os.mkdir(output_path)

    start_end_list = list()

    for process_num in range(CORE) :
        # start_end_list.append([power_list, process_num, START_PROGRAM, process_width, output_time, output_path])
        start_end_list.append([power_list, process_num, int(START_PROGRAM[process_num]), int(END_PROGRAM[process_num]), output_time, output_path])
    
    print(start_end_list)
    p = mp.Pool(CORE)
    p.map(sb.main, start_end_list)
