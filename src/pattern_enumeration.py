import time
import numpy as np
import os
import csv
import datetime


runtime   = time.time()

num_of_beam = 8
TOTAL_POWER = 5
power_list  = []

POWER_GRADATION = 8
power_width     = TOTAL_POWER / POWER_GRADATION

isnewfile  = True
output_csv = False
filename = "database\\decision_power_" + str(POWER_GRADATION) + "_" + datetime.datetime.now().strftime("%m%d-%H%M%S") + ".csv"

def printlen() :
    global power_list

    if len(power_list) > 1 and len(power_list) % 1000 == 0 :
        print(len(power_list), time.time() - runtime)
        print(isnewfile)
        result_output()
        power_list.clear()


def decision_power() :
    def dfs(A):

        if len(A) == num_of_beam:
            # 処理
            if sum(A) == TOTAL_POWER :
                power_list.append(A.copy())
                printlen()
            return
    
        elif len(A) == num_of_beam - 1:
            if sum(A) <= TOTAL_POWER :
                A.append(TOTAL_POWER - sum(A))
                dfs(A)
                A.pop()
        
        else :
            for v in range(POWER_GRADATION+1):
                A.append(power_width * v if v != 0 else 0)
                dfs(A)
                A.pop()

    dfs([])

    print(len(power_list), time.time() - runtime)
    # for _ in power_list :
    #     print(_)

    return power_list


def result_output():
    global isnewfile

    if not output_csv : 
        return

    mode = 'w' if isnewfile == True else 'a'

    with open(filename, mode, newline='') as f:
        writer = csv.writer(f)
        if isnewfile == True : writer.writerow(range(num_of_beam))
        for _ in power_list :
            # print(_)
            writer.writerow(_)
    
    if isnewfile == True : isnewfile = False

power_list = decision_power()
print(isnewfile)
result_output()
