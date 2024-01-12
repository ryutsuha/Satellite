import simulation_bandwidth_mp as sb
import multiprocessing as mp
import time
import datetime
import os
import csv
import math
import pandas as pd
import numpy as np

START_PROGRAM = 10
END_PROGRAM   = 301
PROCESS_WIDTH = 50

num = 180 #km
output_csv = True
result_process_num = list()
power_list_process = list()

now         = datetime.datetime.now()
output_time = now.strftime("%m%d-%H%M%S")
output_path = "Result\\Result_" + output_time

if __name__ == "__main__":

    # power_list = pd.read_csv("database\\decision_power_8_1102-003616.csv", dtype=np.float16)

    if output_csv:
        os.mkdir(output_path)

    start_end_list = list()

    process = math.ceil((END_PROGRAM - START_PROGRAM - 1) / 50)

    print(process)


    for process_num in range(process) :
        start_process_num = START_PROGRAM + process_num * PROCESS_WIDTH 
        end_process_num   = start_process_num + PROCESS_WIDTH if start_process_num + PROCESS_WIDTH < END_PROGRAM else END_PROGRAM
        start_end_list.append([process_num, int(start_process_num), int(end_process_num), output_time, output_path])
    p = mp.Pool(process)
    p.map(sb.main, start_end_list)

    print(start_end_list)
