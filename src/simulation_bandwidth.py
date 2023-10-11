import numpy as np
import pandas as pd
import scipy.special
import matplotlib.pyplot as plt
import csv
import os
import time
import datetime
from geopy.distance import geodesic

runtime   = time.time()
pref_list = pd.read_csv("database\\jinko_list_sityoson.csv")


C                = 299792458  # 光速[m/s]
SAT_DIST         = 35786      # 静止衛星軌道までの距離[km]
TRANS_GAIN       = 52         # dB
RECEIVE_GAIN     = 41         # dB
PROPAGATION_LOSS = -205.5     # dB
DOWNLINK_LOSS    = TRANS_GAIN + RECEIVE_GAIN + PROPAGATION_LOSS  # dB

PLOT_WIDTH       = 91
START_BANDWIDTH  = 2.5 * 10 ** 9    #[Hz]
TOTAL_BANDWIDTH  = 35 * 10 ** 6     #[Hz]
TOTAL_POWER      = 5                #[W]
REPEATED_BEAM    = 3
KAISU            = 100

output_csv       = False
printf           = True
random_param     = False
cni_plot         = False

CENTER  = [30, 128]
NORTH_X = 143
NORTH_Y = 44
CROSS_X = 139
CROSS_Y = 36
SOUTH_X = 130
SOUTH_Y = 33
POINT_X = np.linspace(0, 1800, 91)              # 位置pにspc[0]のポイント数だけサンプルする．
POINT_Y = np.linspace(0, 1800, 91)              # 位置pにspc[1]のポイント数だけサンプルする．
mesh_x, mesh_y = np.meshgrid(POINT_X, POINT_Y)  # np.meshgrid()は，一次元配列2個を使って，座標をメッシュ状に展開する．


if output_csv:
    now = datetime.datetime.now()
    output_time = now.strftime("%m%d-%H%M%S")
    output_path = "Result\\Result_" + output_time
    os.mkdir(output_path)


def result_output_num(iter, cni_mean, beam_bps, beam_cni_db, beam_bps_per_Hz, bps_per_Hz):
    if not output_csv:
        return

    filename = output_path + "\\result_all_" + str(num) + "km_" + output_time + ".csv"
    mode = 'w' if iter == 0 else 'a'

    with open(filename, mode, newline='') as f:
        writer = csv.writer(f)
        header = ["iter", "iter", "beam_num", "band", "power[W]", "bandwidth[Hz]", "beam_cni_db[dB]", "beam_bps_per_Hz", "bitrate[beam_bps]", "latitude", "longitude", "sat_radius", "beam_radius"]
        writer.writerow(header)

        #      [iter, iter, beam_num, band, power[W]  , bandwidth[Hz] , beam_cni_db[dB], beam_bps_per_Hz , bitrate[beam_bps]]
        data = [iter, ""  , ""      , ""  , sum(power), sum(bandwidth), cni_mean       , bps_per_Hz      , sum(beam_bps)    ]
        writer.writerow(data)

        for beam_num in range(num_of_beam):
            freq = determ_freq(freq_beam_list, beam_num)
            #      [iter, iter, beam_num, band, power[W]       , bandwidth[Hz]  , beam_cni_db[dB]      , beam_bps_per_Hz          , bitrate[beam_bps] , latitude                , longitude               , sat_radius          , beam_radius          ]
            data = [""  , iter, beam_num, freq, power[beam_num], bandwidth[freq], beam_cni_db[beam_num], beam_bps_per_Hz[beam_num], beam_bps[beam_num], beam_center[beam_num][0], beam_center[beam_num][1], sat_radius[beam_num], beam_radius[beam_num]]
            writer.writerow(data)

        writer.writerow("")


def result_output(iter, cni_mean, beam_bps, beam_cni_db, beam_bps_per_Hz, bps_per_Hz):
    if not output_csv:
        return

    filename = output_path + "\\result_" + str(num) + "km" + output_time + ".csv"
    mode = 'w' if iter == 0 else 'a'

    with open(filename, mode, newline='') as f:
        writer = csv.writer(f)
        if iter == 0:
            header = ["iter", "power[W]", "bandwidth[Hz]", "cni[dB]", "bps_per_Hz", "bitrate[beam_bps]"]
            writer.writerow(header)

        data = [iter, power, bandwidth, cni_mean, bps_per_Hz, sum(beam_bps)]
        writer.writerow(data)


# 総ビーム数と各ビームの中心座標を返す(指定したビーム配置)
def beam_count():
    for beam_num in range(len(circle)):
        beam_center[beam_num] = ([circle["latitude"][beam_num], circle["longitude"][beam_num]])

    return len(circle)


# 市町村役場所在地から各ビームの中心までの距離を計算し､beam_radius[km]以内ならcenter_dist_listに追加して返す
def pref_beam_distance():

    for pref in range(len(pref_list)): # 市町村の数(沖縄除く)
        beam_center_dist_x = [
            geodesic(beam_center[beam_num], [beam_center[beam_num][0], pref_list['経度'][pref]]).km
            for beam_num in range(num_of_beam)
        ]
        beam_center_dist_y = [
            geodesic(beam_center[beam_num], [pref_list['緯度'][pref], beam_center[beam_num][1]]).km
            for beam_num in range(num_of_beam)
        ]
        beam_center_dist = np.sqrt(np.array(beam_center_dist_x) ** 2 + np.array(beam_center_dist_y) ** 2)
        beam_overlap_list = sum(1 for i, dist in enumerate(beam_center_dist) if dist <= beam_radius[i])

        center_dist_list.extend([
            [pref_list['自治体'][pref], int(pref_list['人口'][pref]), beam_num, beam_overlap_list, x, y]
            for beam_num, x, y in zip(range(num_of_beam), beam_center_dist_x, beam_center_dist_y)
            if beam_center_dist[beam_num] <= beam_radius[beam_num]
        ])

    return center_dist_list


# 各ビームのユーザ数を計算して返す
def user_count():
    pref_user = [center_dist[1] / center_dist[3] for center_dist in center_dist_list]   # 各都道府県の人口を都道府県庁所在地がある場所の範囲内のビームの数だけ割って足す(1ユーザが2ビームで通信されないようにする)
    beam_user = [0] * num_of_beam

    for i, center_dist in enumerate(center_dist_list):
        beam_user[center_dist[2]] += pref_user[i]

    beam_user = [round(user) for user in beam_user]
    return beam_user


def beam_freq():
    freq_beam_list = [[0]] * REPEATED_BEAM

    # for freq_num in range(REPEATED_BEAM):
    #     freq_beam_list.append(list())

    for beam_num in range(num_of_beam):
        this_beam_freq = circle["color"][beam_num] - 1
        freq_beam_list[this_beam_freq].append(beam_num)

    return freq_beam_list


# def initial_power():
#     if random_param:
#         for beam_num in range(num_of_beam):
#             # power[beam_num] = np.random.rand()
#             power.append(np.random.rand())

#         rand_total_power = sum(power)
#         for beam_num in range(num_of_beam):
#             power[beam_num] = power[beam_num] * np.array(TOTAL_POWER / rand_total_power)
#     else:
#         for beam_num in range(num_of_beam):
#             # power[beam_num] = TOTAL_POWER / num_of_beam
#             power.append(TOTAL_POWER / num_of_beam)

#     print("power : ", power, "total : ", sum(power))
#     return power


def initial_bandwidth():
    global bandwidth
    if random_param:
        for freq_num in range(REPEATED_BEAM):
            # bandwidth[freq_num] = np.random.rand()
            bandwidth.append(np.random.rand())

        rand_total_bandwidth = sum(bandwidth)
        for freq_num in range(REPEATED_BEAM):
            bandwidth[freq_num] = bandwidth[freq_num] * np.array(TOTAL_BANDWIDTH / rand_total_bandwidth)
    else:
        bandwidth = [TOTAL_BANDWIDTH / REPEATED_BEAM] * REPEATED_BEAM
        # for freq_num in range(REPEATED_BEAM):
        #     # bandwidth[freq_num] = TOTAL_BANDWIDTH / REPEATED_BEAM
        #     bandwidth.append(TOTAL_BANDWIDTH / REPEATED_BEAM)


    print("bandwidth : ", bandwidth, "total : ", sum(bandwidth))
    return bandwidth


# 各ビームの電力を総当りでリスト化する 今は他のプログラムに投げたので未使用
# 参考 : https://drken1215.hatenablog.com/entry/2020/05/04/190252
# def decision_power() :
#     power_gradation = 32
#     power_width     = TOTAL_POWER / power_gradation
#     def dfs(A):
#         # 数列の長さが beam_su に達したら打ち切り
#         if len(A) == num_of_beam:
            
#             # 処理
#             if sum(A) == TOTAL_POWER :
#                 power_list.append(A.copy())
#             return
        
#         for v in range(power_gradation+1):
#             A.append(power_width * v if v != 0 else v)
#             dfs(A)
#             A.pop()

#     dfs([])

#     print(len(power_list))
#     # for _ in power_list :
#     #     print(_)


def decision_power_from_csv() :
    global power_list
    power_list = pd.read_csv("database\\decision_power_8_0927-115200.csv").values.tolist()
    return power_list
    

def initial_beam_radius():
    for beam_num in range(num_of_beam):
        beam_radius[beam_num] = 60

    return beam_radius


def initial_sat_radius():
    for beam_num in range(num_of_beam):
        sat_radius[beam_num] = circle["sat_radius"][beam_num]  # 衛生アンテナの直径

    return sat_radius


def determ_freq(freq_beam_list, beam_num):
    for freq_num in range(REPEATED_BEAM):
        if beam_num in freq_beam_list[freq_num]:
            # print(f"{beam_num}のビーム番号は{freq_num},周波数帯域幅は{initial_bandwidth()[freq_num]}です")
            return freq_num


def dBm(mW_value):
    return 10 * np.log10(mW_value + 1e-30)  # ゼロ除算を防ぐために1e-30を足す


def mW(dBm_value):
    return 10 ** ((dBm_value)/10)


def add_beam(beam_center, sat_radius, plot):

    # freqField[REPEATED_BEAM][ビーム数][gb, xy][latitude][longitude]
    dist_from_center_x = list()
    dist_from_center_y = list()

    for iter in range(REPEATED_BEAM):
        freqField[iter] = list()

    for beam_num in range(num_of_beam):
        dist_from_center_x.append([geodesic(CENTER, [CENTER[0], beam_center[beam_num][1]]).km])
        dist_from_center_y.append([geodesic(CENTER, [beam_center[beam_num][0], CENTER[1]]).km])
        iter = determ_freq(freq_beam_list, beam_num)
        gb   = beam_gain(START_BANDWIDTH, (dist_from_center_x[beam_num], dist_from_center_y[beam_num]), beam_num, sat_radius)
        freqField[iter].append([gb, [dist_from_center_x[beam_num], dist_from_center_y[beam_num]]])

        for i in range(REPEATED_BEAM):
            if i != iter:
                freqField[i].append([0, [0, 0]])

    if plot:
        # cni_all[beam_num][y][x]
        cni_all = calc_cni(power, bandwidth)[1]

        for beam_num in range(num_of_beam):
            iter = determ_freq(freq_beam_list, beam_num)
            print(cni_all[beam_num], beam_num, iter)
            plot_cni(cni_all[beam_num], beam_num, iter)

    return (dist_from_center_x, dist_from_center_y)


#ビーム利得の算出(G(Theta)の算出)
def beam_gain(freq, dist_from_center, beam_num, sat_radius):  
    lmd    = C / freq                                                   # 波長lambda, 波長=光速/周波数であるから．
    dist_x = dist_from_center[0][0] - mesh_x
    dist_y = dist_from_center[1][0] - mesh_y
    theta  = np.arctan2(np.sqrt(dist_x ** 2 + dist_y ** 2), SAT_DIST)   # 詳しくは中平先生の論文を参照．ビームのゲインを求めるために地上でのxy座標系から曲座標系に変換．
    s      = (np.pi * sat_radius[beam_num]) / lmd * np.sin(theta)       # 詳しくは中平先生の論文を参照．ビームのゲインを求めるために，式の共通項を求める．

    return (2*scipy.special.jv(1, s) / (s + 1e-12)) ** 2                # 詳しくは中平先生の論文を参照．ビームのゲインを求めて返す．scipy.special.jv(n, s)は第nベッセル関数．
                                                                        # 中平先生の論文にはベッセル関数の添字に1があったので，第一次ベッセル関数であると判断して本記述とした．


# 各ビームの範囲内のcni(=cni_beam_avg)と全範囲のcniを返す
def calc_cni(power, bandwidth):
    cni_beam_avg = list()
    cni_all      = list()

    for beam_num in range(num_of_beam):
        cni_in_beam      = list()
        cni_in_beam_dist = list()

        freq_num = determ_freq(freq_beam_list, beam_num)
        carrier = power[beam_num]*1000 * freqField[freq_num][beam_num][0] * mW(DOWNLINK_LOSS)   # freqField[REPEATED_BEAM][beam_num][gb, xy][latitude][longitude]
        noise = 4.3649399999999995e-18 * bandwidth[freq_num]                                    # (1.38 * 10**(-23)) * 316.3 * bandwidth[freq_num] * 1000

        # CI比を知りたいビーム以外のビームをノイズとして加算しまくる．
        interference = sum(
            power[i] * 1000 * freqField[freq_num][i][0] * mW(DOWNLINK_LOSS)
            for i in range(num_of_beam) if i != beam_num
        )

        # points_in_beam[len][x, y, dist]
        points_in_beam = mean_cni(beam_num, dBm(carrier))
        total_cni = carrier / (noise + interference)
        cni_all.append(dBm(total_cni))

        for i in range(len(points_in_beam)):
            cni_in_beam.append(total_cni[points_in_beam[i][0]][points_in_beam[i][1]])
            cni_in_beam_dist.append(points_in_beam[i][2])
        
        beam_radius[beam_num] = max(cni_in_beam_dist)
        cni_beam_avg.append(np.mean(cni_in_beam))

    return (cni_beam_avg, cni_all, beam_radius[beam_num])  # 真値


# ビーム範囲内(ビームの中心からbeam_radius[km]以内)のポイントをリスト化して返す
def mean_cni(beam_num, dBm_C):
    points_in_beam = list()
    beam_edge = np.max(dBm_C) - 3

    dist_x = dist_from_center_x[beam_num][0] - mesh_x
    dist_y = dist_from_center_y[beam_num][0] - mesh_y
    dist   = np.sqrt(dist_x ** 2 + dist_y ** 2)

    x_coords, y_coords = np.where(dBm_C >= beam_edge)   # dBm_Cがbeam_edgeより高ければTrue､低ければFalseを返す(ビーム範囲をTFで返す)
    points_in_beam = [[x, y, dist[x, y]] for x, y in zip(x_coords, y_coords)]

    return points_in_beam


# cniの強弱図を出力
def plot_cni(ci, beam_num, iter):
    #       freqField[REPEATED_BEAM][ビーム数][gb, xy][latitude][longitude]
    point = freqField[iter][beam_num][1]

    BX, BY = np.meshgrid(POINT_X, POINT_Y)
    fig = plt.figure(figsize=(10.24, 7.68))
    ax = fig.add_subplot(111)
    plt.pcolormesh(BX, BY, ci, cmap="gist_ncar")
    plt.colorbar(orientation='vertical')
    x1 = point[0][0] - DOWNLINK_LOSS
    x2 = point[0][0] + DOWNLINK_LOSS
    y1 = point[1][0] - DOWNLINK_LOSS
    y2 = point[1][0] + DOWNLINK_LOSS
    poly = plt.Polygon(((x1, y1), (x1, y2), (x2, y2), (x2, y1)), fill=False)
    ax.add_patch(poly)
    plt.xlabel("POINT_X[km]")
    plt.ylabel("POINT_Y[km]")


def calc_bitrate(cni, iter, printf):

    # 通信方式      [   QPSK1/2, QPSK3/5, QPSK3/4, QPSK5/6, 8PSK2/3, 8PSK5/6, 16APSK3/4, 16APSK8/9, 32APSK4/5, 32APSK9/10]
    # 必要なcni     [0, 1.4    , 2.7    , 4.6    , 5.6    , 7.4    , 10.0   , 11.2     , 13.8     , 15.8     , 19.5      ]
    # 周波数利用効率 [0, 1.00   , 1.20   , 1.50   , 1.67   , 1.98   ,  2.52  ,  3.00    ,  3.50    ,  4.00    , 4.50      ]
    # 上のリストをもとに周波数利用効率の近似線を作成

    beam_bps        = list()
    beam_cni_db     = dBm(np.array(cni))
    cni_mean        = dBm(np.array(cni).mean())
    beam_bps_per_Hz = 0.2179 * beam_cni_db + 0.4387  # 周波数利用効率｡近似線を用いて計算している
    bps_per_Hz      = 0.2179 * cni_mean + 0.4387

    for beam_num in range(num_of_beam):  # 総ビーム数回
        if beam_bps_per_Hz[beam_num] < 0:
            beam_bps_per_Hz[beam_num] = 0

        beam_bps.append(beam_bps_per_Hz[beam_num] * bandwidth[determ_freq(freq_beam_list, beam_num)])

    bps_person_list_num.append([num, freq_beam_list, iter, cni_mean, bps_per_Hz, sum(beam_bps), sum(beam_user)])

    if printf:
        print(f"num {num}, iter: {iter}, cni: {np.round(cni_mean, 12)}[dB], {bps_per_Hz}[bps/Hz], {sum(beam_bps):,}[bps], power : {power} band : {bandwidth}")

        # for beam_num in range(num_of_beam):  # 総ビーム数回
        #     print(f"beam {beam_num}: {np.round(power[beam_num], 6)}[W], cni = {np.round(beam_cni_db[beam_num], 3)}[dB], {beam_user[beam_num]}[人], {beam_bps_per_Hz[beam_num]}[beam_bps/Hz] * {np.round(bandwidth[determ_freq(freq_beam_list, beam_num)]):,}[Hz] = {np.round(beam_bps[beam_num]):,}[beam_bps], {beam_user[beam_num]}[人], {np.round(beam_center[beam_num], 9)}, {sat_radius[beam_num]}[m], {beam_radius[beam_num]}[km]")

        result_output_num(iter, cni_mean, beam_bps, beam_cni_db, beam_bps_per_Hz, bps_per_Hz)
        result_output(iter, cni_mean, beam_bps, beam_cni_db, beam_bps_per_Hz, bps_per_Hz)

    return beam_bps


def calc_bps_max(iter):
    if bps_person_max[5] <= bps_person_list_num[iter][5]:
        return bps_person_list_num[iter]
    else:
        return bps_person_max


if __name__ == '__main__':
    # for num in [180, 120, 60, 45]:
    for num in [180]:

        circle  = pd.read_csv("database\\beam_list_" + str(num) + "km.csv")
        print("database\\beam_list_" + str(num) + "km.csv")

        power               = list()
        bandwidth           = list()
        beam_center         = dict()
        beam_radius         = dict()
        sat_radius          = dict()
        power_list          = list()

        center_dist_list    = list()
        freqField           = dict()   # 角周波数ごとにビームを入れる辞書
        bps_person_max      = [0] * 7
        bps_person_list_num = list()

        num_of_beam         = beam_count()
        freq_beam_list      = beam_freq()

        # initial_power()
        # decision_power()
        decision_power_from_csv()
        initial_bandwidth()
        initial_beam_radius()
        initial_sat_radius()
        pref_beam_distance()

        for iter in range(REPEATED_BEAM):
            if freqField.get(iter) is None:
                freqField[iter] = list()

        dist_from_center   = add_beam(beam_center, sat_radius, plot=False)
        dist_from_center_x = dist_from_center[0]
        dist_from_center_y = dist_from_center[1]

        beam_user = np.round(np.array(user_count()) / 60)
        setup_time = time.time() - runtime
        print(f'開始時間 : {setup_time}')

        # KAISU回学習
        # for iter in range(1):
        for iter in range(len(power_list)):
            power = power_list[iter]

            # freq_user_list = list()
            # beam_user = np.round(np.array(user_count()) / 60)

            # # 同じ周波数で通信しているビームのユーザ数の合計を計算
            # for freq_num in range(REPEATED_BEAM):
            #     freq_user = 0

            #     for beam_num in range(num_of_beam):
            #         if determ_freq(freq_beam_list, beam_num) == freq_num:
            #             freq_user += beam_user[beam_num]

            #     freq_user_list.append(freq_user)

            # dist_from_center   = add_beam(beam_center, sat_radius, plot=cni_plot)
            # dist_from_center_x = dist_from_center[0]
            # dist_from_center_y = dist_from_center[1]

            cni = calc_cni(power, bandwidth)[0]
            calc_bitrate(cni, iter, printf)
            bps_person_max = calc_bps_max(iter)
            # print(f'{iter}回目の学習終わり Process={time.time() - runtime}')

        print(f"実行時間 : {time.time() - runtime - setup_time}")
        print(f"終了時間 : {time.time() - runtime}")
        # result_output()
        print()

plt.show()


# print(len(freqField[0][0][0][0]))  # longitude
# print(len(freqField[0][0][0]))     # latitude
# print(len(freqField[0][0]))        # gb, xy
# print(len(freqField[0]))           # beam_num
# print(len(freqField))              # REPEATED_BEAM

# print(f"freqField[{len(freqField)}][{len(freqField[0])}][{len(freqField[0][0])}][{len(freqField[0][0][0])}][{len(freqField[0][0][0][0])}]")
# freqField[REPEATED_BEAM][ビーム数][gb, xy][latitude][longitude]

# # 都道府県庁所在地から各ビームの中心までの距離を計算し､beam_radius[km]以内ならcenter_dist_listに追加して返す
# def pref_beam_distance_old():
#     center_dist_list   = list()
#     beam_center_dist   = list()
#     beam_center_dist_x = list()
#     beam_center_dist_y = list()
#     beam_overlap_list  = list()

#     for pref in range(len(pref_list)):  # 都道府県の数(沖縄除く)
#         beam_center_dist_x.append(list())
#         beam_center_dist_y.append(list())
#         beam_center_dist.append(list())

#     for pref in range(len(pref_list)):  # 都道府県の数(沖縄除く)
#         beam_overlap = 0

#         for beam_num in range(num_of_beam):
#             # print(beam_num, beam_center[beam_num], [beam_center[beam_num][0]  , pref_list['県庁経度'][pref]])
#             beam_center_dist_x[pref].append(geodesic(beam_center[beam_num], [beam_center[beam_num][0], pref_list['経度'][pref]]).km)
#             beam_center_dist_y[pref].append(geodesic(beam_center[beam_num], [pref_list['緯度'][pref], beam_center[beam_num][1]]).km)
#             beam_center_dist[pref].append(np.sqrt(beam_center_dist_x[pref][beam_num] ** 2 + beam_center_dist_y[pref][beam_num] ** 2))

#             if beam_center_dist[pref][beam_num] <= beam_radius[beam_num]:
#                 beam_overlap += 1

#         beam_overlap_list.append(beam_overlap)

#         for beam_num in range(num_of_beam):
#             if beam_center_dist[pref][beam_num] <= beam_radius[beam_num]:
#                 center_dist_list.append([pref_list['自治体'][pref], int(pref_list['人口'][pref]), beam_num, beam_overlap_list[pref], beam_center_dist_x, beam_center_dist_y])

#     return center_dist_list


# # 各ビームのユーザ数を計算して返す
# def user_count_old():
#     pref_user = list()
#     beam_user = list()

#     for i in range(len(center_dist_list)):
#         # 各都道府県の人口を都道府県庁所在地がある場所の範囲内のビームの数だけ割って足す(1ユーザが2ビームで通信されないようにする)
#         pref_user.append(center_dist_list[i][1] / center_dist_list[i][3])

#     # 各ビームのユーザ数を計算
#     for beam_num in range(num_of_beam):  # 総ビーム数回
#         beam_user.append(0)

#         for i in range(len(center_dist_list)):

#             if beam_num == center_dist_list[i][2]:
#                 beam_user[beam_num] += pref_user[i]

#     for beam_num in range(num_of_beam):  # 総ビーム数回
#         beam_user[beam_num] = round(beam_user[beam_num])
    
#     return beam_user