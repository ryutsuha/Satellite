from adam import adam
import numpy as np
import pandas as pd
import scipy.special
import matplotlib.pyplot as plt
import pprint as pp
import csv
import GPy
import GPyOpt
import time
import multiprocessing as mp
from itertools import combinations
from geopy.distance import geodesic
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process


c = 299792458       # 光速[m/s]
sat_dist = 35786    # 静止衛星軌道までの距離[km]
radius = 20         # 衛星の半径[m]
center = [30, 128]

trans_gain = 52             # dB
receive_gain = 41           # dB
propagation_loss = -205.5   # dB
downlink_loss = trans_gain + receive_gain + propagation_loss # dB

plotwidth = 91
start_bandwidth = 2.5 * 10 ** 9
total_bandwidth = 35 * 10 ** 6
total_power = 12

bandwidth = dict()
power = dict()
num_list = list()


pref_list = pd.read_csv("jinko_list.csv")
point_x = np.linspace(0, 1800, 91)              # 位置pにspc[0]のポイント数だけサンプルする．
point_y = np.linspace(0, 1800, 91)              # 位置pにspc[1]のポイント数だけサンプルする．
mesh_x, mesh_y = np.meshgrid(point_x, point_y)  # np.meshgrid()は，一次元配列2個を使って，座標をメッシュ状に展開する．



# 総ビーム数と各ビームの中心座標を返す(指定したビーム配置)
def beam_count() :
    circle = pd.read_csv("circle.csv")
    beam_center = list()
    
    for i in range(len(circle)):
        beam_center.append([circle["y"][i], circle["x"][i]])

    return beam_center


# 総ビーム数と各ビームの中心座標を返す(ビーム配置も変更する)
def beam_range(rng) :
    circle = pd.read_csv("beam_range.csv")
    beam_center = list()
    
    for i in range(len(circle)):
        beam_center.append([circle["y_min"][i] + rng, circle["x_min"][i] + rng])

    return beam_center



# 都道府県庁所在地から各ビームの中心までの距離を計算し､180km以内ならcenter_dist_listに追加して返す
def pref_beam_distance() :
    beam_overlap_list = list()

    for pref in range(len(pref_list)) : # 都道府県の数(沖縄除く)
        beam_center_dist_x.append(list())
        beam_center_dist_y.append(list())
        beam_center_dist.append(list())

    for pref in range(len(pref_list)) : # 都道府県の数(沖縄除く)
        beam_overlap = 0

        for beam_num in range(num_of_beam) :
            beam_center_dist_x[pref].append(geodesic(beam_center[beam_num], [beam_center[beam_num][0]  , pref_list['県庁経度'][pref]]).km)
            beam_center_dist_y[pref].append(geodesic(beam_center[beam_num], [pref_list['県庁緯度'][pref], beam_center[beam_num][1]]).km)
            beam_center_dist[pref].append(np.sqrt(beam_center_dist_x[pref][beam_num] ** 2 + beam_center_dist_y[pref][beam_num] ** 2))

            if beam_center_dist[pref][beam_num] <= 180 :
                beam_overlap +=1

            # ビーム0の範囲内に道庁所在地(札幌市)が含まれていないため例外処理
            if pref_list['都道府県'][pref] == '北海道' and beam_num == 0 : 
                beam_overlap += 1
        
        beam_overlap_list.append(beam_overlap)

        for beam_num in range(num_of_beam) :
            
            if beam_center_dist[pref][beam_num] <= 180 :
                center_dist_list.append([pref_list['都道府県'][pref], pref_list['人口'][pref],  beam_num, beam_overlap_list[pref], beam_center_dist_x, beam_center_dist_y])

            # ビーム0の範囲内に道庁所在地(札幌市)が含まれていないため例外処理
            if pref_list['都道府県'][pref] == '北海道' and beam_num == 0 : 
                center_dist_list.append([pref_list['都道府県'][pref], pref_list['人口'][pref],  beam_num, beam_overlap_list[pref], beam_center_dist_x, beam_center_dist_y])

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

    print(beam_user)
    return beam_user


# 周波数とビームの組み合わせリストを返す
def beam_combinations() :
    combination = list()
    first = list(range(num_of_beam))


    for i in range(int(num_of_beam / 2)) :
        combination.extend(list(combinations(first, i + 1)))

    for i in range(len(combination)) :
        num_list.append([combination[i]])

    for i in range(len(num_list)) :
        num_list[i][0] = list(num_list[i][0])
        
        for j in range(repeated_beam - 1) :
            num_list[i].append(list())

        for k in range(num_of_beam) :
            if k not in num_list[i][0] :
                num_list[i][1].append(k)
    

def initial_bandwidth() :
    for freq_num in range(repeated_beam) :
        # bandwidth.append(total_bandwidth / repeated_beam)
        bandwidth[freq_num] = total_bandwidth / repeated_beam
    
    return bandwidth


def initial_power() :
    for beam_num in range(num_of_beam) :
        # power.append(total_power / num_of_beam)
        power[beam_num] = total_power / num_of_beam

    return power


def determ_freq(num_list, beam_num) :
    for repeated in range(repeated_beam) :
        if beam_num in num_list[repeated] :
            # print(f"{beam_num}のビーム番号は{repeated},周波数帯域幅は{initial_bandwidth()[repeated]}です")
            return repeated


def dBm(mW_value) :
    return 10 * np.log10(mW_value + 1e-30) # ゼロ除算を防ぐために1e-30を足す


def mW(dBm_value) :
    return 10 ** ((dBm_value)/10)    


def add_beam(num, plot) :
    for beam_num in range(num_of_beam) :
        dist_from_center_x.append(list())
        dist_from_center_y.append(list())

    for beam_num in range(num_of_beam) :
        dist_from_center_x[beam_num].append(geodesic(center, [center[0], beam_center[beam_num][1]]).km)
        dist_from_center_y[beam_num].append(geodesic(center, [beam_center[beam_num][0], center[1]]).km)
        iter = determ_freq(num_list[num], beam_num)
        # print(f"beam: {beam_num}, freq: {iter}, center: {beam_center[beam_num]}, x: {dist_from_center_x[beam_num]}, y: {dist_from_center_y[beam_num]} km")

        gb = beam_gain(start_bandwidth, (dist_from_center_x[beam_num], dist_from_center_y[beam_num]))
        freqField[iter].append([gb, [dist_from_center_x[beam_num], dist_from_center_y[beam_num]]])

        for i in range(repeated_beam) :
            if i != iter : 
                freqField[i].append([0, [0, 0]])
    
    if plot :
        # CNI_all[beam_num][y][x]
        CNI_all = calc_CNI(power, bandwidth)[1]
    
        for beam_num in range(num_of_beam) :
            iter = determ_freq(num_list[num], beam_num)
            plot_CNI(CNI_all[beam_num], beam_num, iter)
    

#ビーム利得の算出(G(Theta)の算出)
def beam_gain(freq, dist_from_center):  
    lmd    = c / freq                                                   # 波長lambda, 波長=光速/周波数であるから．
    dist_x = dist_from_center[0][0] - mesh_x
    dist_y = dist_from_center[1][0] - mesh_y
    theta  = np.arctan2(np.sqrt(dist_x ** 2 + dist_y ** 2), sat_dist)   # 詳しくは中平先生の論文を参照．ビームのゲインを求めるために地上でのxy座標系から曲座標系に変換．
    s      = (np.pi * radius) / lmd * np.sin(theta)                     # 詳しくは中平先生の論文を参照．ビームのゲインを求めるために，式の共通項を求める．

    return (2*scipy.special.jv(1, s) / (s + 1e-12)) ** 2                # 詳しくは中平先生の論文を参照．ビームのゲインを求めて返す．scipy.special.jv(n, s)は第nベッセル関数．
                                                                        # 中平先生の論文にはベッセル関数の添字に1があったので，第一次ベッセル関数であると判断して本記述とした．


# ビーム範囲内(ビームの中心から180km以内)のポイントをリスト化して返す
def mean_CNI(beam_num) :
    dist_x = dist_from_center_x[beam_num][0] - mesh_x
    dist_y = dist_from_center_y[beam_num][0] - mesh_y
    dist   = np.sqrt(dist_x ** 2 + dist_y ** 2)

    points_in_beam = list()

    for x in range(len(dist)):
        for y in range(len(dist)):
            if dist[x][y] <= 180 :
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


# 各ビームの範囲内のCNI(=dbb)と全範囲のCNIを返す
def calc_CNI(power, bandwidth):
    dbb = list()
    CNI_all = list()

    for beam_num in range(num_of_beam) :
        iter = determ_freq(num_list[num], beam_num)
        # freqField[repeated_beam][beam_num][gb, xy][latitude][longitude]
        C = power[beam_num]*1000 * freqField[iter][beam_num][0] * mW(downlink_loss)
        N = (1.38 * 10**(-23)) * 316.3 * bandwidth[determ_freq(num_list[num], beam_num)] * 1000
        I = np.zeros_like(C)

        # ビーム数回繰り返す
        for i in range(num_of_beam) :
            if i != beam_num:
                this = power[i]*1000 * freqField[iter][i][0] * mW(downlink_loss)
                # CI比を知りたいビーム以外のビームをノイズとして加算しまくる．
                I += this
        
        points_in_beam = mean_CNI(beam_num) # points_in_beam[len][x, y, dist]
        CNI_in_beam = list()

        for i in range(len(points_in_beam)) :
            CNI_in_beam.append((C / (N + I))[points_in_beam[i][0]][points_in_beam[i][1]])
        
        dbb.append(np.array(CNI_in_beam).mean())
        CNI_all.append(dBm(C / (N + I)))
    
    # return dBm(C / (N + I))
    return (dbb, CNI_all) # 真値


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


def bitrate(efficiency, beam_num) :
    return efficiency * bandwidth[determ_freq(num_list[num], beam_num)]


#微分
def df_power(params):
    params_dx = params.copy()
    dx = 1e-0

    for beam_num in range(num_of_beam) : # 総ビーム数回
        params_dx[beam_num] += dx
    
    before_CNI = np.array(calc_CNI(params, bandwidth)[0])
    after_CNI = np.array(calc_CNI(params_dx, bandwidth)[0])

    before_CNI = np.array(calc_bitrate(before_CNI, iter, kaisu, printf = False))
    after_CNI = np.array(calc_bitrate(after_CNI, iter, kaisu, printf = False))
    CNI_dx = (after_CNI - before_CNI) / dx

    return CNI_dx


def df_bandwidth(params):
    params_dx = params.copy()
    dx = 10 ** 6

    for beam_num in range(num_of_beam) : # 総ビーム数回
        params_dx[determ_freq(num_list[num], beam_num)] += dx
    
    before_CNI = np.array(calc_CNI(power, params)[0])
    after_CNI = np.array(calc_CNI(power, params_dx)[0])

    before_CNI = np.array(calc_bitrate(before_CNI, iter, kaisu, printf = False))
    after_CNI = np.array(calc_bitrate(after_CNI, iter, kaisu, printf = False))
    CNI_dx = abs((after_CNI - before_CNI) * dx)

    return CNI_dx


def calc_bitrate(CNI, iter, kaisu, printf) :
    beam_CNI = dBm(np.array(CNI))
    CNI_mean = dBm(np.array(CNI).mean())
    bps = list()
    bps_Hz = list()
    bps_person = list()
    bps_person_list_num = list()
    sowa = 0

    for beam_num in range(num_of_beam) : # 総ビーム数回
        bps_Hz.append(spectrum_efficiency(beam_CNI[beam_num]))
        bps.append(bitrate(bps_Hz[beam_num], beam_num))
        if beam_user[beam_num] == 0 :
            bps_person.append(0)
        else :
            bps_person.append(bitrate(bps_Hz[beam_num], beam_num) / beam_user[beam_num])
        sowa += bitrate(bps_Hz[beam_num], beam_num)

    if printf:
        if iter == kaisu-1 :
        # if True :
            print(f"num {num}, iter: {iter}, CNI: {CNI_mean}[dB], {spectrum_efficiency(CNI_mean)}[bps/Hz]")

        # if iter == kaisu-1 :
        if False :
            for beam_num in range(num_of_beam) : # 総ビーム数回
                print(f"beam {beam_num}: {np.round(power[beam_num], 6)}[W], CNI = {np.round(beam_CNI[beam_num], 3)}[dB], {bps_Hz[beam_num]}[bps/Hz] * {np.round(bandwidth[determ_freq(num_list[num], beam_num)]):,}[Hz] = {np.round(bps[beam_num]):,}[bps], {beam_user[beam_num]}[人], {np.round(bps_person[beam_num]):,}[bps/人]")

            print(f"sowa : {sowa:,}")
    
        bps_person_list_num.append([iter, CNI_mean, spectrum_efficiency(CNI_mean), sowa, np.average(bps_person)])

        for i in range(len(bps_person_list_num)) :
            if bps_person_max[6] <= bps_person_list_num[i][4] :
                bps_person_max[0] = num
                bps_person_max[1] = num_list[num]
                bps_person_max[2] = bps_person_list_num[i][0]
                bps_person_max[3] = bps_person_list_num[i][1]
                bps_person_max[4] = bps_person_list_num[i][2]
                bps_person_max[5] = bps_person_list_num[i][3]
                bps_person_max[6] = bps_person_list_num[i][4]
            # print()

        result_output_num(iter, CNI_mean, bps, bps_person, sowa, beam_CNI, bps_Hz)

    return bps_person


def result_output_num(iter, CNI_mean, bps, bps_person, sowa, beam_CNI, bps_Hz) :
    if output_csv :

        if iter == 0 :
            filename = 'result\\' + str(num) + ' ' + str(num_list[num]) + '.csv'
            f = open(filename, 'w', newline='')
            writer = csv.writer(f)
            data = ["iter", "iter", "beam_num", "user", "power[W]", "bandwidth[Hz]", "beam_CNI[dB]", "bps_Hz", "bitrate[bps]", "bps/user", "total bitrate"]
            writer.writerow(data)

        else :
            filename = 'result\\' + str(num) + ' ' + str(num_list[num]) + '.csv'
            f = open(filename, 'a', newline='')
            writer = csv.writer(f)

        data = [iter, "", "", sum(beam_user), sum(power.values()), sum(bandwidth.values()), CNI_mean, spectrum_efficiency(CNI_mean), np.average(bps), sowa / sum(beam_user), sowa]
        writer.writerow(data)

        for beam_num in range(num_of_beam) :
            data = ["", iter, beam_num, beam_user[beam_num], power[beam_num], bandwidth[determ_freq(num_list[num], beam_num)], beam_CNI[beam_num], bps_Hz[beam_num], bps[beam_num], (bps_person[beam_num])]
            writer.writerow(data)

        writer.writerow("")
        f.close()


def result_output() :
    if output_csv :

        if num == 0 :
            f = open('result\\result.csv', 'w', newline='')
            writer = csv.writer(f)
            data = ["", "num_list", "iter", "CNI", "bps/Hz", "bps", "bps/man"]
            writer.writerow(data)

        else :
            f = open('result\\result.csv', 'a', newline='')
            writer = csv.writer(f)
        
        data = bps_person_list[num]
        writer.writerow(data)
        f.close


def main() :
    for iter in range(repeated_beam) :
        if freqField.get(iter) is None:
            freqField[iter] = list()

    add_beam(num, plot = False)
    
    # 同じ周波数で通信しているビームのユーザ数の合計を計算
    for freq_num in range(repeated_beam) :
        freq_user = 0

        for beam_num in range(num_of_beam) :
            if determ_freq(num_list[num], beam_num) == freq_num :
                freq_user += beam_user[beam_num]

        freq_user_list.append(freq_user)


    for iter in range(kaisu) :
        grads_power = df_power(power)
        grads_bandwidth = df_bandwidth(bandwidth)
        
        power_opt.update(power, grads_power, total_power, beam_user, "power")
        bandwidth_opt.update(bandwidth, grads_bandwidth, total_bandwidth, beam_user, freq_user_list)

        CNI = calc_CNI(power, bandwidth)[0]
        calc_bitrate(CNI, iter, kaisu, printf)
    
    bps_person_list.append(bps_person_max)
    # print(bps_person_list[num])

    result_output()



if __name__ == '__main__':

    repeated_beam = 2
    num_of_beam = 8
    kaisu = 2
    output_csv = False
    printf = True
    t = time.time()
    beam_combinations()
    # num_list = [[[0, 2, 4, 6], [1, 3, 5, 7]]]
    # num_list = [[[0], [1, 2, 3, 4, 5, 6, 7]]]
    # num_list = [[[0], [1], [2], [3], [4], [5], [6], [7]]]

    # for rng in range(0, 31, 10):
    for rng in range(1):

        beam_center = beam_range(rng/10)
        print(rng/10, beam_center)

        center_dist_list = list()
        beam_center_dist_x = list()
        beam_center_dist_y = list()
        beam_center_dist = list()
        dist_from_center_x = list()
        dist_from_center_y = list()
        bps_person_list = list()

        pref_beam_distance() 
        beam_user = np.round(np.array(user_count()) / 60)


        process_list = list()

        for num in range(len(num_list)) :
            
            freqField = dict()               # 角周波数ごとにビームを入れる辞書
            freq_user_list = list()
            bps_person_max = [0, 0, 0, 0, 0, 0, 0]
            power_opt = adam()
            bandwidth_opt = adam()
            
            initial_power()
            initial_bandwidth()

            mp.freeze_support()
            # main()
            # executor.submit(main(num))

        #     arg = (repeated_beam, freqField, num, beam_center, num_list, beam_user, freq_user_list, kaisu, power_opt, bandwidth_opt, power, bandwidth, total_bandwidth)
        #     process_list.append(Process(target=main, args=arg))
        
        # for num in range(len(num_list)) :
        
        #     process_list[num].start()
        #     print(num, "is Started")
        #     process_list[num].join()


        # for num in range(len(num_list)) :


    print(f'Process={time.time() - t}')






# print(len(freqField[0][0][0][0]))  # longitude
# print(len(freqField[0][0][0]))     # latitude
# print(len(freqField[0][0]))        # gb, xy
# print(len(freqField[0]))           # beam_num
# print(len(freqField))              # repeated_beam

# print(f"freqField[{len(freqField)}][{len(freqField[0])}][{len(freqField[0][0])}][{len(freqField[0][0][0])}][{len(freqField[0][0][0][0])}]")
# freqField[repeated_beam][ビーム数][gb, xy][latitude][longitude]
plt.show()
