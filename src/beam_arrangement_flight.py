import xlrd
import shapely
import time
import csv
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as tick # 目盛り操作に必要なライブラリを読み込みます
import matplotlib.patches as patches
from geopy.distance import geodesic


t = time.time()
color  = list()
colorlist = {"cl0" : "#888888", "cl1" : "#4FAAD1", "cl2" : "#EBBF00", "cl3" : "#B66427", "cl4" : "#0f4047"}

beam_center = list()
beam_param  = list()
beam_radius = list()
beam_freq   = list()
beam_center_output_tf = list()


def setup () :
    for lat in range(len(latitude)) :
        for lon in range(len(longitude)) :
            if lat % 2 == 0 :
                beam_center[i].append([latitude[lat], longitude[lon]])
            
            else :
                beam_center[i].append([latitude[lat], longitude[lon] + beam_dist * 0.15])
            
            # freq_num = (lon + i) % 3
            # beam_freq.append(freq_num)
            beam_radius.append(sat_rad)

    print(f"lat: {len(latitude)}, lon: {len(longitude)}")


def beam_append() :

    for pt in range(len(beam_center[i])) :
        border_lat = beam_center[i][pt][1] * inclination[i] + section[i]
        # print(beam_center[i][pt][1], border_lat)


        if beam_center[i][pt][0] > border_lat - 0.05 and beam_center[i][pt][0] < border_lat + 0.05 :
            beam_center_output_tf.append([pt, beam_center[i][pt][1], beam_center[i][pt][0], True])

        else :
            beam_center_output_tf.append([pt, beam_center[i][pt][1], beam_center[i][pt][0], False])


def appen_beam_param() :
    if   i == 0 : adjst = 0
    elif i == 1 : adjst = 0
    elif i == 2 : adjst = 1
    elif i == 3 : adjst = 2
    elif i == 4 : adjst = 2
    elif i == 5 : adjst = 0
    else : adjst = 0

    for pt in range(len(beam_center[i])) :

        if beam_center_output_tf[pt][3] is True :
            freq_num = (pt+adjst) % 3 + 1
            beam_freq.append(freq_num)
            beam_param.append([beam_center[i][pt][0], beam_center[i][pt][1], beam_radius[pt], beam_freq[-1], i])
            # print(pt,    beam_center[i][pt][0], beam_center[i][pt][1], beam_radius[pt], beam_freq[pt])


def output_pic() :
    fig, ax = plt.subplots(figsize = (10,10))
    jpmap = gpd.read_file('japan.geojson')
    jpmap.plot(ax = ax, edgecolor='#444', facecolor='white', linewidth = 0.5, aspect="equal")

    for i in range(len(inclination)) :
        plt.axline((0, section[i]), slope=inclination[i], color='blue')

    for i in range(len(beam_center)) :
        for pt in range(len(beam_center[i])) :
            if beam_center_output_tf[pt][3] is True :
                plt.scatter(beam_center[i][pt][1], beam_center[i][pt][0], marker="+", color="black")

    plt.xlim([122, 147])
    plt.ylim([20, 47])
    plt.gca().xaxis.set_minor_locator(tick.MultipleLocator(1))
    plt.gca().yaxis.set_minor_locator(tick.MultipleLocator(1))

    for beam_num in range(len(beam_param)):
        ax.add_patch(patches.Circle(xy=(beam_param[beam_num][1], beam_param[beam_num][0]), radius=beam_param[beam_num][2]/100, color=colorlist["cl" + str(beam_param[beam_num][3])], alpha=0.225))


def output_csv () :
    for beam_num in range(len(beam_param)):
        filename = 'beam_list.csv'

        if beam_num == 0 :
            f = open(filename, 'w', newline='')
            writer = csv.writer(f)
            data = ["latitude", "longitude", "beam_radius", "color", "i"]
            writer.writerow(data)

        else :
            f = open(filename, 'a', newline='')
            writer = csv.writer(f)
        
        data = beam_param[beam_num]
        writer.writerow(data)

inclination  = [0.590917135     , 0.641637878     , 0.301275519     ] * 2
section_base = [-49.1874        , -52.23          , -5.725          ]
section      = [-49.1874 -0.4   , -52.23 -0.4     , -5.725 -0.4     , -49.1874 +0.4   , -52.23 +0.4     , -5.725 +0.4]
sin_lon      = [0.50814886197829, 0.54024711965688, 0.28734788556635] * 2
cos_lat      = [0.86126925759032, 0.84150641691103, 0.95782628522115] * 2



sat_rad = 45
beam_dist = (sat_rad / 15)


for i in range(len(inclination)) :
    beam_center.append(list())
    longitude = np.array(list(np.arange(1220, 1465, beam_dist)))/10
    latitude = list()

    for j in range(len(longitude)) :
        latitude.append(longitude[j] * inclination[i] + section[i])


    setup()
    beam_append()
    appen_beam_param()

output_pic()
output_csv()

print(f'Process={time.time() - t}')
plt.show()
