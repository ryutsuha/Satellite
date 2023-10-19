import simulation_bandwidth_mp as sb
import multiprocessing as mp
import time
import datetime
import os
import csv
import pandas as pd

CORE = 12
num = 180 #km
output_csv = True
result_process_num = list()

if __name__ == "__main__":

    if output_csv:
        now         = datetime.datetime.now()
        output_time = now.strftime("%m%d-%H%M%S")
        output_path = "Result\\Result_" + output_time
        os.mkdir(output_path)

    start_end_list = list()
    power_list_num = len(sb.power_list)
    process_width = power_list_num / CORE

    for process_num in range(CORE) :
        start_end_list.append([int(process_num * process_width), int((process_num+1)*process_width), process_num, output_time, output_path])
    
    p = mp.Pool(CORE)
    p.map(sb.main, start_end_list)
