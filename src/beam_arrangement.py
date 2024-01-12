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
df = gpd.read_file('japan_eez.json')


beam_center = list()
beam_param = list()
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


# 都道府県庁所在地から各ビームの中心までの距離を計算し､beam_radius[km]以内ならcenter_dist_listに追加して返す
def pref_beam_distance() :
    center_dist_list = list()

    for pref in range(len(pref_list)): # 市町村の数(沖縄除く)
        print("pref_beam_distance", pref)
        beam_center_dist_x = [
            geodesic(beam_center[beam_num], [beam_center[beam_num][0], pref_list['経度'][pref]]).km
            for beam_num in range(len(beam_center))
        ]
        beam_center_dist_y = [
            geodesic(beam_center[beam_num], [pref_list['緯度'][pref], beam_center[beam_num][1]]).km
            for beam_num in range(len(beam_center))
        ]
        beam_center_dist = np.sqrt(np.array(beam_center_dist_x) ** 2 + np.array(beam_center_dist_y) ** 2)
        beam_overlap_list = sum(1 for i, dist in enumerate(beam_center_dist) if dist <= beam_radius[i])

        center_dist_list.extend([
            [pref_list['自治体'][pref], int(pref_list['人口'][pref]), beam_num, beam_overlap_list, x, y]
            for beam_num, x, y in zip(range(len(beam_center)), beam_center_dist_x, beam_center_dist_y)
            if beam_center_dist[beam_num] <= beam_radius[beam_num]
        ])


    print("pref_beam_distanceおわり")
    return center_dist_list


# 各ビームのユーザ数を計算して返す
def user_count() :
    pref_user = list()
    beam_user = list()

    for i in range(len(center_dist_list)):
        # 各都道府県の人口を都道府県庁所在地がある場所の範囲内のビームの数だけ割って足す(1ユーザが2ビームで通信されないようにする)
        pref_user.append(center_dist_list[i][1] / center_dist_list[i][3]) 

    # 各ビームのユーザ数を計算
    for beam_num in range(len(beam_center)) : # 総ビーム数回
        beam_user.append(0)

        for i in range(len(center_dist_list)):
        
            if beam_num == center_dist_list[i][2] :
                beam_user[beam_num] += pref_user[i]
    
    for beam_num in range(len(beam_center)) : # 総ビーム数回
        beam_user[beam_num] = round(beam_user[beam_num])
        print(beam_num, beam_user[beam_num])
    
    print("user_count終わり")
    return beam_user


def appen_beam_param() :
    for beam_num in range(len(beam_center)) :
        if user[beam_num] != 0 :
            beam_param.append([beam_center[beam_num][0], beam_center[beam_num][1], beam_radius[beam_num], beam_freq[beam_num], user[beam_num]])
            print(beam_num,    beam_center[beam_num][0], beam_center[beam_num][1], beam_radius[beam_num], beam_freq[beam_num], user[beam_num])


def output_pic() :
    fig, ax = plt.subplots(figsize = (10,10))
    # df = gpd.read_file('japan.geojson')
    df.plot(ax = ax, edgecolor='#444', facecolor='white', linewidth = 0.5, aspect="equal")

    plt.scatter(128, 30, marker="+", color="black")
    plt.scatter(146.676333, 30, marker="+", color="black")
    plt.scatter(128, 46.216165, marker="+", color="black")
    plt.scatter(146.676333, 46.216165, marker="+", color="black")

    plt.xlim([128, 146.676333])
    plt.ylim([30, 46.216165])
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

sat_rad_list = [120]

for _ in range(len(sat_rad_list)) :

    sat_rad = sat_rad_list[_]
    beam_dist = int(sat_rad / 7.5)
    longitude = np.array(list(range(1280, 1465, beam_dist)))/10
    latitude = np.array(list(range(300, 460, beam_dist)))/10

    setup()
    center_dist_list = pref_beam_distance()
    user = user_count()
    # user = [1] * len(beam_center)
    appen_beam_param()

output_pic()
output_csv()

print(f'Process={time.time() - t}')
plt.show()
