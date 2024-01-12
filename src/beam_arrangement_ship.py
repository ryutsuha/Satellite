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
pref_list = pd.read_csv("database\\jinko_list_sityoson.csv")
# pref_list = pd.read_csv("database\\jinko_list_sityoson_ships_airlines.csv")
eez = gpd.read_file('japan_eez.json')
hoppou = gpd.read_file('japan_hoppou.json')
dev_zone = gpd.read_file('japan_korea_dev_zone.json')
senkaku = gpd.read_file('japan_senkaku.json')

beam_center = list()
beam_param  = list()
beam_radius = list()
beam_freq   = list()


def setup () :
    for lat in range(len(latitude)) :
        for lon in range(len(longitude)) :
            if lat % 2 == 0 :
                beam_center.append([latitude[lat], longitude[lon]])
            
            else :
                beam_center.append([latitude[lat], longitude[lon] + beam_dist * 0.15])
            
            beam_freq.append(lon % 3 + 1)
            beam_radius.append(sat_rad)

    print(f"lat: {len(latitude)}, lon: {len(longitude)}")


def beam_eez_io() :
    beam_center_shape = list()
    beam_center_eez_tf = list()
    poly_eez = eez.iloc[0].geometry
    poly_dev_zone = dev_zone.iloc[0].geometry
    poly_senkaku = senkaku.iloc[0].geometry

    for pt in range(len(beam_center)) :
        beam_center_shape.append(shapely.geometry.point.Point(beam_center[pt][1], beam_center[pt][0]))
        
        if any([poly_eez.contains(beam_center_shape[pt]), poly_dev_zone.contains(beam_center_shape[pt]), poly_senkaku.contains(beam_center_shape[pt])]) :
            beam_border = 7.5/11 * beam_center[pt][1] -63.7
            
            if (beam_center[pt][0] > beam_border) :
                beam_center_eez_tf.append([pt, beam_center[pt][1], beam_center[pt][0], True])
            
            else :
                beam_center_eez_tf.append([pt, beam_center[pt][1], beam_center[pt][0], False])
        
        else :
            beam_center_eez_tf.append([pt, beam_center[pt][1], beam_center[pt][0], False])

    return beam_center_eez_tf


def appen_beam_param() :
    for pt in range(len(beam_center)) :
        if beam_center_eez_tf[pt][3] is True :
            beam_param.append([beam_center[pt][0], beam_center[pt][1], beam_radius[pt], beam_freq[pt]])
            print(pt,    beam_center[pt][0], beam_center[pt][1], beam_radius[pt], beam_freq[pt])


def output_pic() :
    fig, ax = plt.subplots(figsize = (10,10))
    # eez = gpd.read_file('japan.geojson')
    eez.plot(ax = ax, edgecolor='#444', facecolor='white', linewidth = 0.5, aspect="equal")
    hoppou.plot(ax = ax, edgecolor='#444', facecolor='white', linewidth = 0.5, aspect="equal")
    dev_zone.plot(ax = ax, edgecolor='#444', facecolor='white', linewidth = 0.5, aspect="equal")
    senkaku.plot(ax = ax, edgecolor='#444', facecolor='white', linewidth = 0.5, aspect="equal")

    # plt.scatter(128, 30, marker="+", color="black")
    # plt.scatter(146.676333, 30, marker="+", color="black")
    # plt.scatter(128, 46.216165, marker="+", color="black")
    # plt.scatter(146.676333, 46.216165, marker="+", color="black")
    plt.axline((0, -63.7), slope=7.5/11, color='black')

    for pt in range(len(beam_center)) :
        if beam_center_eez_tf[pt][3] is True :
            plt.scatter(beam_center[pt][1], beam_center[pt][0], marker="+", color="black")

    plt.xlim([122, 147])
    plt.ylim([20, 47])
    plt.gca().xaxis.set_minor_locator(tick.MultipleLocator(1))
    plt.gca().yaxis.set_minor_locator(tick.MultipleLocator(1))

    for beam_num in range(len(beam_param)):
        ax.add_patch(patches.Circle(xy=(beam_param[beam_num][1], beam_param[beam_num][0]), radius=beam_param[beam_num][2]/100, color=colorlist["cl" + str(beam_param[beam_num][3])], alpha=0.225))


def output_csv () :
    for beam_num in range(len(beam_param)):
        # ax.add_patch(patches.Circle(xy=(beam_center[beam_num][1], beam_center[beam_num][0]), radius=beam_radius[beam_num]/100, color=colorlist["cl" + str(beam_freq[beam_num])], alpha=0.3))
        filename = 'beam_list.csv'

        if beam_num == 0 :
            f = open(filename, 'w', newline='')
            writer = csv.writer(f)
            data = ["latitude", "longitude", "beam_radius", "color", "user"]
            writer.writerow(data)

        else :
            f = open(filename, 'a', newline='')
            writer = csv.writer(f)
        
        data = beam_param[beam_num]
        writer.writerow(data)

sat_rad_list = [45]

for _ in range(len(sat_rad_list)) :

    sat_rad = sat_rad_list[_]
    beam_dist = int(sat_rad / 7.5)
    longitude = np.array(list(range(1220, 1465, beam_dist)))/10
    latitude = np.array(list(range(210, 460, beam_dist)))/10

    setup()

    beam_center_eez_tf = beam_eez_io()
    appen_beam_param()

output_pic()
output_csv()

print(f'Process={time.time() - t}')
plt.show()
