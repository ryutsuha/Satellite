from adam import adam
import numpy as np
import pandas as pd
import scipy.special
import matplotlib.pyplot as plt
import pprint as pp
import csv
import os
import time
import datetime
import math
import keras
import multiprocessing as mp
import tensorflow as tf
from itertools import combinations
from geopy.distance import geodesic
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process
from matplotlib import path

runtime  = time.time()
filename = os.path.splitext(os.path.basename(__file__))[0] #ファイル名を取得
filename = filename.split() # ビーム半径､ 開始回数(num)

start_num = int(filename[1])
pref_list = pd.read_csv("database\\jinko_list_sityoson.csv")
circle  = pd.read_csv("database\\beam_list_" + filename[0] + "km.csv")

if int(filename[0]) == 180 :
    end_num = start_num + 100

elif int(filename[0]) == 120 :
    end_num = start_num + 50

elif int(filename[0]) == 60 :
    end_num = start_num + 25

elif int(filename[0]) == 45 :
    end_num = start_num + 15 if start_num + 15 > 100 else 100  # パラメータのランダム化が100回を超えないようにする


c        = 299792458  # 光速[m/s]
sat_dist = 35786      # 静止衛星軌道までの距離[km]
trans_gain       = 52     # dB
receive_gain     = 41     # dB
propagation_loss = -205.5 # dB
downlink_loss    = trans_gain + receive_gain + propagation_loss # dB

plotwidth       = 91
start_bandwidth = 2.5 * 10 ** 9
total_bandwidth = 35 * 10 ** 6
total_power     = 5
repeated_beam   = 3
kaisu           = 100

output_csv      = True
printf          = True
random_param    = True

center   = [30, 128]
north_x = 143
north_y = 44
cross_x = 139
cross_y = 36
south_x = 130
south_y = 33
point_x = np.linspace(0, 1800, 91)              # 位置pにspc[0]のポイント数だけサンプルする．
point_y = np.linspace(0, 1800, 91)              # 位置pにspc[1]のポイント数だけサンプルする．
mesh_x, mesh_y = np.meshgrid(point_x, point_y)  # np.meshgrid()は，一次元配列2個を使って，座標をメッシュ状に展開する．

if output_csv :
    now         = datetime.datetime.now()
    output_time = now.strftime("%m%d-%H%M%S")
    output_path = "Result\\Result_" + output_time
    os.mkdir(output_path)


def result_output_num(iter, CNI_mean, bps, bps_person, sowa, beam_CNI, bps_Hz) :
    if output_csv :
        filename = output_path + "\\result_all_" + str(num) + ".csv"

        if iter == 0 :
            f = open(filename, 'w', newline='')

        else :
            f = open(filename, 'a', newline='')

        writer = csv.writer(f)
        data = ["iter", "iter", "beam_num", "user", "power[W]", "bandwidth[Hz]", "beam_CNI[dB]", "bps_Hz", "bitrate[bps]", "bps/user", "latitude", "longitude", "sat_radius", "beam_radius", "freq"]
        writer.writerow(data)
        #      [iter, iter, beam_num, user          , power[W]           , bandwidth[Hz]          , beam_CNI[dB], bps_Hz                       , bitrate[bps]   , bps/user             ]
        data = [iter, ""  , ""      , sum(beam_user), sum(power.values()), sum(bandwidth.values()), CNI_mean    , spectrum_efficiency(CNI_mean), sowa           , sowa / sum(beam_user)]
        writer.writerow(data)

        for beam_num in range(num_of_beam) :
            #      [iter, iter, beam_num, user               , power[W]       , bandwidth[Hz]                                  , beam_CNI[dB]      , bps_Hz          , bitrate[bps] , bps/user            , latitude                , longitude               , sat_radius          , beam_radius          , freq]
            data = [""  , iter, beam_num, beam_user[beam_num], power[beam_num], bandwidth[determ_freq(num_list, beam_num)], beam_CNI[beam_num], bps_Hz[beam_num], bps[beam_num], bps_person[beam_num], beam_center[beam_num][0], beam_center[beam_num][1], sat_radius[beam_num], beam_radius[beam_num], circle["color"][beam_num] - 1]
            writer.writerow(data)

        writer.writerow("")
        f.close()


def result_output() :
    if output_csv :
        filename = output_path + "\\result_" + str(num_of_beam) + "beam.csv"

        if num - start_num == 0 :
            f = open(filename, 'w', newline='')
            writer = csv.writer(f)
            data = ["", "num_list", "iter", "CNI", "bps/Hz", "bps", "user", "bps/man"]
            writer.writerow(data)

        else :
            f = open(filename, 'a', newline='')
            writer = csv.writer(f)
        
        # data = bps_person_list[num - start_num]
        data = bps_person_max
        writer.writerow(data)
        f.close


# 総ビーム数と各ビームの中心座標を返す(指定したビーム配置)
def beam_count() :
    for beam_num in range(len(circle)):
        beam_center[beam_num] = ([circle["latitude"][beam_num], circle["longitude"][beam_num]])

    return len(circle)


# 都道府県庁所在地から各ビームの中心までの距離を計算し､beam_radius[km]以内ならcenter_dist_listに追加して返す
def pref_beam_distance() :
    center_dist_list = list()
    beam_center_dist = list()
    beam_center_dist_x = list()
    beam_center_dist_y = list()
    beam_overlap_list = list()

    for pref in range(len(pref_list)) : # 都道府県の数(沖縄除く)
        beam_center_dist_x.append(list())
        beam_center_dist_y.append(list())
        beam_center_dist.append(list())

    for pref in range(len(pref_list)) : # 都道府県の数(沖縄除く)
        beam_overlap = 0

        for beam_num in range(num_of_beam) :
            # print(beam_num, beam_center[beam_num], [beam_center[beam_num][0]  , pref_list['県庁経度'][pref]])
            beam_center_dist_x[pref].append(geodesic(beam_center[beam_num], [beam_center[beam_num][0]  , pref_list['経度'][pref]]).km)
            beam_center_dist_y[pref].append(geodesic(beam_center[beam_num], [pref_list['緯度'][pref], beam_center[beam_num][1]]).km)
            beam_center_dist[pref].append(np.sqrt(beam_center_dist_x[pref][beam_num] ** 2 + beam_center_dist_y[pref][beam_num] ** 2))

            if beam_center_dist[pref][beam_num] <= beam_radius[beam_num] :
                beam_overlap +=1

        beam_overlap_list.append(beam_overlap)

        for beam_num in range(num_of_beam) :
            
            if beam_center_dist[pref][beam_num] <= beam_radius[beam_num] :
                center_dist_list.append([pref_list['自治体'][pref], int(pref_list['人口'][pref]),  beam_num, beam_overlap_list[pref], beam_center_dist_x, beam_center_dist_y])

    return center_dist_list


# 各ビームのユーザ数を計算して返す
def user_count() :
    pref_user = list()
    beam_user = list()

    for i in range(len(center_dist_list)):
        # 各都道府県の人口を都道府県庁所在地がある場所の範囲内のビームの数だけ割って足す(1ユーザが2ビームで通信されないようにする)
        pref_user.append(center_dist_list[i][1] / center_dist_list[i][3]) 

    # 各ビームのユーザ数を計算
    for beam_num in range(num_of_beam) : # 総ビーム数回
        beam_user.append(0)

        for i in range(len(center_dist_list)):
        
            if beam_num == center_dist_list[i][2] :
                beam_user[beam_num] += pref_user[i]
    
    for beam_num in range(num_of_beam) : # 総ビーム数回
        beam_user[beam_num] = round(beam_user[beam_num])
    return beam_user


def beam_freq() :
    num_list = list()

    for freq_num in range(repeated_beam) :
        num_list.append(list())

    for beam_num in range(num_of_beam) :
        this_beam_freq = circle["color"][beam_num] - 1
        num_list[this_beam_freq].append(beam_num)

    return num_list
    

def initial_power() :
    if random_param :
        for beam_num in range(num_of_beam) :
            power[beam_num] = np.random.rand()
        
        rand_total_power = sum(power.values())
        
        for beam_num in range(num_of_beam) :
            power[beam_num] = power[beam_num] * np.array(total_power / rand_total_power)
    
    else :
        for beam_num in range(num_of_beam) :
            power[beam_num] = total_power / num_of_beam

    # print("power : ", power, "total : ", sum(power.values()))
    return power


def initial_bandwidth() :
    if random_param :
        for freq_num in range(repeated_beam) :
            bandwidth[freq_num] = np.random.rand()
        
        rand_total_bandwidth = sum(bandwidth.values())
    
        for freq_num in range(repeated_beam) :
            bandwidth[freq_num] = bandwidth[freq_num] * np.array(total_bandwidth / rand_total_bandwidth)
    
    else :
        for freq_num in range(repeated_beam) :
            bandwidth[freq_num] = total_bandwidth / repeated_beam
    
    # print("bandwidth : ", bandwidth, "total : ", sum(bandwidth.values()))
    return bandwidth


def initial_beam_radius() :
    for beam_num in range(num_of_beam) :
        beam_radius[beam_num] = 60

    return beam_radius


def initial_sat_radius() :
    for beam_num in range(num_of_beam) :
        sat_radius[beam_num] = circle["sat_radius"][beam_num] # 衛生アンテナの直径

    return sat_radius


def determ_freq(num_list, beam_num) :
    for repeated in range(repeated_beam) :
        if beam_num in num_list[repeated] :
            # print(f"{beam_num}のビーム番号は{repeated},周波数帯域幅は{initial_bandwidth()[repeated]}です")
            return repeated


def dBm(mW_value) :
    return 10 * np.log10(mW_value + 1e-18) # ゼロ除算を防ぐために1e-30を足す


def mW(dBm_value) :
    return 10 ** ((dBm_value)/10)    


def add_beam(beam_center, sat_radius, plot) :

    for iter in range(repeated_beam) :
        freqField[iter] = list()

    # freqField[repeated_beam][ビーム数][gb, xy][latitude][longitude]
    dist_from_center_x = list()
    dist_from_center_y = list()

    for beam_num in range(num_of_beam) :
        dist_from_center_x.append(list())
        dist_from_center_y.append(list())

    for beam_num in range(num_of_beam) :
        dist_from_center_x[beam_num].append(geodesic(center, [center[0], beam_center[beam_num][1]]).km)
        dist_from_center_y[beam_num].append(geodesic(center, [beam_center[beam_num][0], center[1]]).km)
        iter = determ_freq(num_list, beam_num)

        gb = beam_gain(start_bandwidth, (dist_from_center_x[beam_num], dist_from_center_y[beam_num]), beam_num, sat_radius)
        freqField[iter].append([gb, [dist_from_center_x[beam_num], dist_from_center_y[beam_num]]])

        for i in range(repeated_beam) :
            if i != iter : 
                freqField[i].append([0, [0, 0]])
        
    if plot :
        # CNI_all[beam_num][y][x]
        CNI_all = calc_CNI(power, bandwidth)[1]
    
        for beam_num in range(num_of_beam) :
            iter = determ_freq(num_list, beam_num)
            plot_CNI(CNI_all[beam_num], beam_num, iter)

    return (dist_from_center_x, dist_from_center_y)
    

#ビーム利得の算出(G(Theta)の算出)
def beam_gain(freq, dist_from_center, beam_num, sat_radius):  
    lmd    = c / freq                                                   # 波長lambda, 波長=光速/周波数であるから．
    dist_x = dist_from_center[0][0] - mesh_x
    dist_y = dist_from_center[1][0] - mesh_y
    theta  = np.arctan2(np.sqrt(dist_x ** 2 + dist_y ** 2), sat_dist)   # 詳しくは中平先生の論文を参照．ビームのゲインを求めるために地上でのxy座標系から曲座標系に変換．
    s      = (np.pi * sat_radius[beam_num]) / lmd * np.sin(theta)       # 詳しくは中平先生の論文を参照．ビームのゲインを求めるために，式の共通項を求める．

    return (2*scipy.special.jv(1, s) / (s + 1e-12)) ** 2                # 詳しくは中平先生の論文を参照．ビームのゲインを求めて返す．scipy.special.jv(n, s)は第nベッセル関数．
                                                                        # 中平先生の論文にはベッセル関数の添字に1があったので，第一次ベッセル関数であると判断して本記述とした．


# 各ビームの範囲内のCNI(=dbb)と全範囲のCNIを返す
def calc_CNI(power, bandwidth):
    dbb = list()
    CNI_all = list()

    for beam_num in range(num_of_beam) :
        iter = determ_freq(num_list, beam_num)
        # freqField[repeated_beam][beam_num][gb, xy][latitude][longitude]
        C = power[beam_num]*1000 * freqField[iter][beam_num][0] * mW(downlink_loss)
        N = (1.38 * 10**(-23)) * 316.3 * bandwidth[iter] * 1000
        I = np.zeros_like(C)

        # ビーム数回繰り返す
        for i in range(num_of_beam) :
            if i != beam_num:
                this = power[i]*1000 * freqField[iter][i][0] * mW(downlink_loss)
                I += this   # CI比を知りたいビーム以外のビームをノイズとして加算しまくる．
        
        dBm_C = dBm(C)
        points_in_beam = mean_CNI(beam_num, dBm_C) # points_in_beam[len][x, y, dist]
        CNI_in_beam = list()
        CNI_in_beam_dist = list()

        for i in range(len(points_in_beam)) :
            CNI_in_beam.append((C / (N + I))[points_in_beam[i][0]][points_in_beam[i][1]])
            CNI_in_beam_dist.append(points_in_beam[i][2])

        beam_radius[beam_num] = max(CNI_in_beam_dist)
        dbb.append(np.array(CNI_in_beam).mean())
        CNI_all.append(dBm(C / (N + I)))

    return (dbb, CNI_all, beam_radius[beam_num]) # 真値


# ビーム範囲内(ビームの中心からbeam_radius[km]以内)のポイントをリスト化して返す
def mean_CNI(beam_num, dBm_C) :
    max_C = list()
    points_in_beam = list()

    for i in range(len(dBm_C)) :
        max_C.append(max(dBm_C[i]))
    beam_edge = max(max_C) - 3

    dist_x = dist_from_center_x[beam_num][0] - mesh_x
    dist_y = dist_from_center_y[beam_num][0] - mesh_y
    dist   = np.sqrt(dist_x ** 2 + dist_y ** 2)

    for x in range(len(dBm_C)):
        for y in range(len(dBm_C[x])):
            if dBm_C[x][y] >= beam_edge :
                points_in_beam.append([x, y, dist[x][y]])
    
    return points_in_beam


# CNIの強弱図を出力
def plot_CNI(ci, beam_num, iter):
    #       freqField[repeated_beam][ビーム数][gb, xy][latitude][longitude]
    point = freqField[iter][beam_num][1]
            
    BX, BY = np.meshgrid(point_x,point_y)
    fig = plt.figure(figsize = (10.24, 7.68))
    ax = fig.add_subplot(111)
    plt.pcolormesh(BX, BY, ci, cmap="gist_ncar")
    plt.colorbar(orientation='vertical')
    x1 = point[0][0] - downlink_loss
    x2 = point[0][0] + downlink_loss
    y1 = point[1][0] - downlink_loss
    y2 = point[1][0] + downlink_loss
    poly = plt.Polygon(((x1, y1), (x1, y2), (x2, y2), (x2, y1)), fill = False)
    ax.add_patch(poly)
    plt.xlabel("point_x[km]")
    plt.ylabel("point_y[km]")


# CNIを元に何bps/Hzで通信可能かを返す
def spectrum_efficiency(CNI) :
    required_cni  = [0, 1.4 , 2.7 , 4.6 , 5.6 , 7.4 , 10.0 , 11.2 , 13.8 , 15.8 , 19.5 ]
    efficiency    = [0, 1.00, 1.20, 1.50, 1.67, 1.98,  2.52,  3.00,  3.50,  4.00,  4.50]
    usable_method = 0

    # required_cniとCNIの差を出し､0以下なら通信可能なのでusable_methodにそのrequired_cniの値を入れる｡
    for comp in required_cni :
        if comp - CNI <= 0 :
            usable_method = comp

    # usable_methodが入ってるrequired_cniの配列番号を探し､その番号のefficiencyを返す
    return efficiency[required_cni.index(usable_method)]


def bitrate(bps_Hz, beam_num) :
    return bps_Hz * bandwidth[determ_freq(num_list, beam_num)]


#微分
def df_power(params):
    params_dx = params.copy()
    dx = 1e-0
    
    before_CNI = np.array(calc_CNI(params, bandwidth)[0])

    for beam_num in range(num_of_beam) : # 総ビーム数回
        params_dx[beam_num] += dx

    add_beam(beam_center, sat_radius, plot = False)
    after_CNI = np.array(calc_CNI(params_dx, bandwidth)[0])
    CNI_dx = (after_CNI - before_CNI) / dx

    return CNI_dx 


def df_bandwidth(params):
    params_dx = params.copy()
    # dx = 10 ** 6
    dx = 1e-0
    
    before_CNI = np.array(calc_CNI(power, params)[0])

    for beam_num in range(num_of_beam) : # 総ビーム数回
        params_dx[determ_freq(num_list, beam_num)] += dx

    add_beam(beam_center, sat_radius, plot = False)
    after_CNI = np.array(calc_CNI(power, params_dx)[0])
    CNI_dx = (after_CNI - before_CNI) / dx

    return CNI_dx 


def df_beam_x(params):
    params_dx = params.copy()
    dx = 1e-3
    
    before_CNI = np.array(calc_CNI(power, bandwidth)[0])

    for beam_num in range(num_of_beam) : # 総ビーム数回
        params_dx[beam_num][1] += dx

    add_beam(params_dx, sat_radius, plot = False)
    after_CNI = np.array(calc_CNI(power, bandwidth)[0])
    CNI_dx = (after_CNI - before_CNI) / dx

    return CNI_dx 


def df_beam_y(params):
    params_dx = params.copy()
    dx = 1e-3

    before_CNI = np.array(calc_CNI(power, bandwidth)[0])

    for beam_num in range(num_of_beam) : # 総ビーム数回
        params_dx[beam_num][0] += dx

    add_beam(params_dx, sat_radius, plot = False)
    after_CNI = np.array(calc_CNI(power, bandwidth)[0])
    CNI_dx = (after_CNI - before_CNI) / dx

    return CNI_dx 


def df_sat_radius(params):
    params_dx = params.copy()
    dx = 1e-1

    before_CNI = np.array(calc_CNI(power, bandwidth)[0])

    for beam_num in range(num_of_beam) : # 総ビーム数回
        params_dx[beam_num] += dx

    add_beam(beam_center, params_dx, plot = False)
    after_CNI = np.array(calc_CNI(power, bandwidth)[0])
    CNI_dx = (after_CNI - before_CNI) / dx

    return CNI_dx


def calc_bitrate(CNI, iter, printf) :
    beam_CNI            = dBm(np.array(CNI))
    CNI_mean            = dBm(np.array(CNI).mean())
    bps                 = list()
    bps_Hz              = list()
    bps_person          = list()
    sowa                = 0

    for beam_num in range(num_of_beam) : # 総ビーム数回
        bps_Hz.append(spectrum_efficiency(beam_CNI[beam_num]))
        bps.append(bitrate(bps_Hz[beam_num], beam_num))
        if beam_user[beam_num] == 0 :
            bps_person.append(0)
        else :
            bps_person.append(bitrate(bps_Hz[beam_num], beam_num) / beam_user[beam_num])
        sowa += bitrate(bps_Hz[beam_num], beam_num)

    if printf:
        print(f"num {num}, iter: {iter}, CNI: {np.round(CNI_mean, 3)}[dB], {spectrum_efficiency(CNI_mean)}[bps/Hz], sowa : {sowa:,}, user : {sum(beam_user)}[人], avg bitrate : {sowa / sum(beam_user):,}")

        # for beam_num in range(num_of_beam) : # 総ビーム数回
            # print(f"beam {beam_num}: {np.round(power[beam_num], 6)}[W], CNI = {np.round(beam_CNI[beam_num], 3)}[dB], {beam_user[beam_num]}[人], {bps_Hz[beam_num]}[bps/Hz] * {np.round(bandwidth[determ_freq(num_list, beam_num)]):,}[Hz] = {np.round(bps[beam_num]):,}[bps], {beam_user[beam_num]}[人], {np.round(bps_person[beam_num]):,}[bps/人], {np.round(beam_center[beam_num], 9)}, {sat_radius[beam_num]}[m], {beam_radius[beam_num]}[km]")

    bps_person_list_num.append([num, num_list, iter, CNI_mean, spectrum_efficiency(CNI_mean), sowa, sum(beam_user),  sowa / sum(beam_user)])
    result_output_num(iter, CNI_mean, bps, bps_person, sowa, beam_CNI, bps_Hz)

    return bps_person


def calc_bps_max(iter) :
    if bps_person_max[5] <= bps_person_list_num[iter][5] :
        return bps_person_list_num[iter]
    else :
        return bps_person_max


# 未使用
def points_in_range() :

    for beam_num in range(num_of_beam) :
        polygon = path.Path([[45, 142], [45, 144], [43, 144], [35, 140], [32, 131], [32, 129], [34, 129], [37, 138]])
        polygon2 = [[45, 142], [45, 144], [43, 144], [35, 140], [32, 131], [32, 129], [34, 129], [37, 138]]

        if polygon.contains_point(beam_center[beam_num]) is True :
            print(f"{beam_center[beam_num]}は{polygon.contains_point(beam_center[beam_num])}")
            True

        else :
            distance = list()
            for i in range(len(polygon)):
                j = (i + 1) % len(polygon)
                distance.append(math.sqrt((polygon2[i][0] - beam_center[beam_num][0]) ** 2 + (polygon2[i][1] - beam_center[beam_num][1]) ** 2))

            print(f"{beam_center[beam_num]}はFalseなので{polygon2[distance.index(min(distance))]}")


if __name__ == '__main__':
    for num in range(start_num, end_num, 1) :
        
        power            = dict()
        bandwidth        = dict()
        beam_center      = dict()
        beam_radius      = dict()
        sat_radius       = dict()

        # bps_person_list  = list()
        center_dist_list = list()
        freqField        = dict()   # 角周波数ごとにビームを入れる辞書
        bps_person_max   = [0] * 8
        bps_person_list_num = list()

        power_opt        = adam()
        bandwidth_opt    = adam()
        beam_opt_x       = adam()
        beam_opt_y       = adam()
        radius_opt       = adam()

        num_of_beam = beam_count()
        num_list    = beam_freq()

        initial_power()
        initial_bandwidth()
        initial_beam_radius()
        initial_sat_radius()

        for iter in range(repeated_beam) :
            if freqField.get(iter) is None:
                freqField[iter] = list()

        dist_from_center   = add_beam(beam_center, sat_radius, plot = False)
        dist_from_center_x = dist_from_center[0]
        dist_from_center_y = dist_from_center[1]
        print(f'今からcenter_dist_list Process={time.time() - runtime}')
        center_dist_list   = pref_beam_distance()
        print(f'セットアップ終わり Process={time.time() - runtime}')

        # kaisu回学習
        for iter in range(kaisu+1) :

            freq_user_list = list()
            beam_user = np.round(np.array(user_count()) / 60)

            # 同じ周波数で通信しているビームのユーザ数の合計を計算
            for freq_num in range(repeated_beam) :
                freq_user = 0

                for beam_num in range(num_of_beam) :
                    if determ_freq(num_list, beam_num) == freq_num :
                        freq_user += beam_user[beam_num]

                freq_user_list.append(freq_user)

            dist_from_center   = add_beam(beam_center, sat_radius, plot = False)
            dist_from_center_x = dist_from_center[0]
            dist_from_center_y = dist_from_center[1]

            if iter != 0 :
                grads_power        = df_power(power)
                power_opt.update_power(power, grads_power, total_power, beam_user, "power")

                grads_bandwidth    = df_bandwidth(bandwidth)
                bandwidth_opt.update_bandwidth(bandwidth, grads_bandwidth, total_bandwidth, beam_user, freq_user_list)
            
                grads_beam_x       = df_beam_x(beam_center)
                beam_opt_x.update_beam(beam_center, grads_beam_x)

                grads_beam_y       = df_beam_y(beam_center)
                beam_opt_y.update_beam(beam_center, grads_beam_y)

                grads_sat_radius  = df_sat_radius(sat_radius)
                radius_opt.update_sat_radius(sat_radius, grads_sat_radius)

            CNI = calc_CNI(power, bandwidth)[0]
            calc_bitrate(CNI, iter, printf)
            bps_person_max = calc_bps_max(iter)
            print(f'{iter}回目の学習終わり Process={time.time() - runtime}')

        print(f'Process={time.time() - runtime}')
        result_output()
        print()


# print(len(freqField[0][0][0][0]))  # longitude
# print(len(freqField[0][0][0]))     # latitude
# print(len(freqField[0][0]))        # gb, xy
# print(len(freqField[0]))           # beam_num
# print(len(freqField))              # repeated_beam

# print(f"freqField[{len(freqField)}][{len(freqField[0])}][{len(freqField[0][0])}][{len(freqField[0][0][0])}][{len(freqField[0][0][0][0])}]")
# freqField[repeated_beam][ビーム数][gb, xy][latitude][longitude]
plt.show()
