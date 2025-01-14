import numpy as np
import pandas as pd
import scipy.special
import matplotlib.pyplot as plt
import csv
import os
import time
import datetime
import geopandas as gpd
from geopy.distance import geodesic

runtime   = time.time()
# circle  = pd.read_csv("database\\beam_list_180km.csv")


circlelist = ["database\\beam_list_ship_180.csv", 
              "database\\beam_list_ship_120.csv",
              "database\\beam_list_ship_60.csv",
              "database\\beam_list_ship_45.csv",
              "database\\beam_list_flight_180.csv",
              "database\\beam_list_flight_120.csv",
              "database\\beam_list_flight_60.csv",
              "database\\beam_list_flight_45.csv"]

outputcirclelist = ["ship_180km", 
                    "ship_120km", 
                    "ship_60km", 
                    "ship_45km", 
                    "air_180km", 
                    "air_120km", 
                    "air_60km", 
                    "air_45km"]


currentlist = 4

circle  = pd.read_csv(circlelist[currentlist])



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
cni_plot         = True

# CENTER  = [30, 128]
CENTER  = [20, 122]
NORTH_X = 143
NORTH_Y = 44
CROSS_X = 139
CROSS_Y = 36
SOUTH_X = 130
SOUTH_Y = 33
POINT_X = np.linspace(0, 3000, 151)              # 位置pにspc[0]のポイント数だけサンプルする．
POINT_Y = np.linspace(0, 3000, 151)              # 位置pにspc[1]のポイント数だけサンプルする．
mesh_x, mesh_y = np.meshgrid(POINT_X, POINT_Y)  # np.meshgrid()は，一次元配列2個を使って，座標をメッシュ状に展開する．

output_time = 0
output_path = 0
process_num = 0


def result_output_num(iter, cni_mean, beam_bps, beam_cni_db, beam_bps_per_Hz, bps_per_Hz):
    if not output_csv:
        return

    filename = output_path + "\\result_all_" + str(outputcirclelist[currentlist]) + "_" + output_time + "_" + str(process_num) +".csv"
    # filename = output_path + "\\result_all_" + str(num_of_beam) + "beam_" + output_time + "_" + str(process_num) +".csv"
    mode = 'w' if iter == start_sat_radius else 'a'

    with open(filename, mode, newline='') as f:
        writer = csv.writer(f)
        header = ["iter", "iter", "beam_num", "band", "power[W]", "bandwidth[Hz]", "beam_cni_db[dB]", "beam_bps_per_Hz", "bitrate[beam_bps]", "latitude", "longitude", "sat_radius", "beam_radius", "carrier", "interference"]
        writer.writerow(header)

        #      ["iter", "iter", "beam_num", "band", "power[W]", "bandwidth[Hz]", "beam_cni_db[dB]", "beam_bps_per_Hz", "bitrate[beam_bps]", "latitude", "longitude", "sat_radius", "beam_radius", "carrier"                , "interference"]
        data = [iter  , ""    , ""        , ""    , sum(power), sum(bandwidth) , cni_mean         , bps_per_Hz       , sum(beam_bps)      , ""        , ""         , ""          , ""           , dBm(np.mean(cni_data[3])), dBm(np.mean(cni_data[4]))]
        writer.writerow(data)

        for beam_num in range(num_of_beam):
            freq_num = determ_freq(freq_beam_list, beam_num)
            #      [iter, iter, beam_num, freq_num, power[W]       , bandwidth[Hz]      , beam_cni_db[dB]      , beam_bps_per_Hz          , bitrate[beam_bps] , latitude                , longitude               , sat_radius          , beam_radius          ,  carrier                   ,interference]
            data = [""  , iter, beam_num, freq_num, power[beam_num], bandwidth[freq_num], beam_cni_db[beam_num], beam_bps_per_Hz[beam_num], beam_bps[beam_num], beam_center[beam_num][0], beam_center[beam_num][1], sat_radius[beam_num], cni_data[2][beam_num],  dBm(cni_data[3][beam_num]), dBm(cni_data[4][beam_num])]
            writer.writerow(data)

        writer.writerow("")


def result_output(iter, cni_mean, beam_bps, beam_cni_db, beam_bps_per_Hz, bps_per_Hz):
    if not output_csv:
        return

    filename = output_path + "\\result_" + str(num_of_beam) + "beam_" + output_time + "_" + str(process_num) +".csv"
    mode = 'w' if iter == start_sat_radius else 'a'

    with open(filename, mode, newline='') as f:
        writer = csv.writer(f)
        if iter == start_sat_radius:
            header = ["iter", "power[W]", "bandwidth[Hz]", "cni[dB]", "bps_per_Hz", "bitrate[beam_bps]", "user"]
            writer.writerow(header)

        data = [iter, power, bandwidth, cni_mean, bps_per_Hz, sum(beam_bps)]
        writer.writerow(data)


# 総ビーム数と各ビームの中心座標を返す(指定したビーム配置)
def beam_count():
    beam_center = list()
    for beam_num in range(len(circle)):
        beam_center.append([circle["latitude"][beam_num], circle["longitude"][beam_num]])

    return len(circle), beam_center


def beam_freq():
    freq_beam_list = list()

    for freq_num in range(REPEATED_BEAM):
        freq_beam_list.append(list())

    for beam_num in range(num_of_beam):
        this_beam_freq = circle["color"][beam_num] - 1
        # freq_beam_list[this_beam_freq].append(beam_num)
        freq_beam_list[this_beam_freq].append(beam_num)

    return freq_beam_list


def initial_power():
    global power
    if random_param:
        for beam_num in range(num_of_beam):
            # power[beam_num] = np.random.rand()
            power.append(np.random.rand())

        rand_total_power = sum(power)
        for beam_num in range(num_of_beam):
            power[beam_num] = power[beam_num] * np.array(TOTAL_POWER / rand_total_power)
    else:
        power = [TOTAL_POWER / num_of_beam] * num_of_beam

    return power


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

    return bandwidth


def initial_sat_radius():
    sat_radius = list()
    for beam_num in range(num_of_beam):
        sat_radius.append(circle["sat_radius"][beam_num])  # 衛生アンテナの直径

    return sat_radius


def determ_freq(freq_beam_list, beam_num):
    for freq_num in range(REPEATED_BEAM):
        if beam_num in freq_beam_list[freq_num]:
            # print(f"{beam_num}のビーム番号は{freq_num},周波数帯域幅は{initial_bandwidth()[freq_num]}です")
            return freq_num


def dBm(mW_value):
    return 10 * np.log10(mW_value + 1e-12)  # ゼロ除算を防ぐために1e-30を足す


def mW(dBm_value):
    return 10 ** ((dBm_value)/10)


def add_beam(beam_center, sat_radius):

    # freqField[REPEATED_BEAM][ビーム数][0=gb]
    # freqField[REPEATED_BEAM][ビーム数][1=xy][longitude][latitude]
    dist_from_center_x = list()
    dist_from_center_y = list()

    for iter in range(REPEATED_BEAM):
        freqField[iter] = list()

    for beam_num in range(num_of_beam):
        dist_from_center_x.append([geodesic(CENTER, [CENTER[0], beam_center[beam_num][1]]).km]) #longitude
        dist_from_center_y.append([geodesic(CENTER, [beam_center[beam_num][0], CENTER[1]]).km]) #latitude
        freq_num = determ_freq(freq_beam_list, beam_num)
        gb   = beam_gain(START_BANDWIDTH, (dist_from_center_x[beam_num], dist_from_center_y[beam_num]), beam_num, sat_radius)
        freqField[freq_num].append([gb, [dist_from_center_x[beam_num], dist_from_center_y[beam_num]]])

        for i in range(REPEATED_BEAM):
            if i != freq_num:
                freqField[i].append([0, [dist_from_center_x[beam_num], dist_from_center_y[beam_num]]])

    return (dist_from_center_x, dist_from_center_y)


#ビーム利得の算出(G(Theta)の算出)
def beam_gain(freq, dist_from_center, beam_num, sat_radius):  
    lmd    = C / freq                                                   # 波長lambda, 波長=光速/周波数であるから．
    dist_x = dist_from_center[0][0] - mesh_x
    dist_y = dist_from_center[1][0] - mesh_y
    theta  = np.arctan2(np.sqrt(dist_x ** 2 + dist_y ** 2), SAT_DIST)   # 詳しくは中平先生の論文を参照．ビームのゲインを求めるために地上でのxy座標系から曲座標系に変換．
    s      = (np.pi * sat_radius[beam_num]) / lmd * np.sin(theta)       # 詳しくは中平先生の論文を参照．ビームのゲインを求めるために，式の共通項を求める．

    return (2*scipy.special.jv(1, s) / (s + 1e-12)) ** 2                # 詳しくは中平先生の論文を参照．ビームのゲインを求めて返す．scipy.special.jv(this_sat_radius, s)は第nベッセル関数．
                                                                        # 中平先生の論文にはベッセル関数の添字に1があったので，第一次ベッセル関数であると判断して本記述とした．


# 各ビームの範囲内のcni(=cni_beam_avg)と全範囲のcniを返す
def calc_cni(power, bandwidth):
    cni_beam_avg = list()
    cni_all_dB      = list()
    beam_radius_def  = list()
    carrier          = list()
    carrier_in_beam = list()
    carrier_all_db = list()
    interference     = list()
    interference_in_beam = list()
    interference_all_db = list()

    for beam_num in range(num_of_beam):
        cni_in_beam      = list()
        cni_in_beam_dist = list()
        carrier_this_beam     = list()
        interference_this_beam     = list()

        freq_num = determ_freq(freq_beam_list, beam_num)
        carrier.append(power[beam_num]*1000 * freqField[freq_num][beam_num][0] * dl_mw)   # freqField[REPEATED_BEAM][beam_num][gb]
        noise = 4.3649399999999995e-18 * bandwidth[freq_num]                                    # (1.38 * 10**(-23)) * 316.3 * bandwidth[freq_num] * 1000

        # CI比を知りたいビーム以外のビームを干渉電力として加算しまくる．
        interference.append(sum(
            power[i] * 1000 * freqField[freq_num][i][0] * dl_mw
            for i in range(num_of_beam) if i != beam_num
        ))

        # points_in_beam[len][x, y, dist]
        points_in_beam = in_beam_tf(beam_num, dBm(carrier[beam_num]))
        cni_all = carrier[beam_num] / (noise + interference[beam_num])
        cni_all_dB.append(dBm(cni_all))
        carrier_all_db.append(dBm(carrier[beam_num]))
        interference_all_db.append(dBm(interference[beam_num]))

        print(    noise ,     np.mean(interference[beam_num]) ,     np.mean(noise + interference[beam_num]),
              dBm(noise), dBm(np.mean(interference[beam_num])), dBm(np.mean(noise + interference[beam_num])))


        for i in range(len(points_in_beam)):
            cni_in_beam.append(cni_all[points_in_beam[i][0]][points_in_beam[i][1]])
            cni_in_beam_dist.append(points_in_beam[i][2])
            carrier_this_beam.append(carrier[beam_num][points_in_beam[i][0]][points_in_beam[i][1]])
            interference_this_beam.append(interference[beam_num][points_in_beam[i][0]][points_in_beam[i][1]])

        beam_radius_def.append(max(cni_in_beam_dist))
        cni_beam_avg.append(np.mean(cni_in_beam))
        carrier_in_beam.append(sum(carrier_this_beam))
        interference_in_beam.append(sum(interference_this_beam))

    return (cni_beam_avg, cni_all_dB, beam_radius_def, carrier_in_beam, interference_in_beam, carrier_all_db, interference_all_db)  # 真数, 対数, 真数, 真数, 真数

# ビーム範囲内(ビームの中心からbeam_radius[km]以内)のポイントをリスト化して返す
def in_beam_tf(beam_num, dBm_C):
    points_in_beam = list()
    beam_edge = np.max(dBm_C) - 3

    dist_x = dist_from_center[0][beam_num][0] - mesh_x
    dist_y = dist_from_center[1][beam_num][0] - mesh_y
    dist   = np.sqrt(dist_x ** 2 + dist_y ** 2)

    x_coords, y_coords = np.where(dBm_C >= beam_edge)   # dBm_Cがbeam_edgeより高ければTrue､低ければFalseを返す(ビーム範囲をTFで返す)
    points_in_beam = [[x, y, dist[x, y]] for x, y in zip(x_coords, y_coords)]

    return points_in_beam


# cniの強弱図を出力
def plot_cni():
    df = gpd.read_file('japan.geojson')

    for _ in range(1):
        beam_num = 2
    # for beam_num in range(num_of_beam):
        BX, BY = np.meshgrid(POINT_X, POINT_Y)
        fig = plt.figure(figsize=(10.24, 7.68))
        ax1 = fig.add_subplot(111)
        plt.pcolormesh(BX, BY, cni_data[6][beam_num], cmap="CMRmap")
        plt.colorbar(orientation='vertical')
        plt.clim(-120, -100)
        # plt.clim(-20, 10)
        df.plot(ax = ax1, edgecolor='#444', facecolor='white', linewidth = 0.5, aspect="equal")

        plt.xlabel("POINT_X[km]")
        plt.ylabel("POINT_Y[km]")


def calc_bitrate(cni, iter, this_sat_radius):

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
        if beam_bps_per_Hz[beam_num] < 0: beam_bps_per_Hz[beam_num] = 0
        beam_bps.append(beam_bps_per_Hz[beam_num] * bandwidth[determ_freq(freq_beam_list, beam_num)])

    bps_person_list_num.append([this_sat_radius, freq_beam_list, iter, cni_mean, bps_per_Hz, sum(beam_bps), ])
    result_output_num(iter, cni_mean, beam_bps, beam_cni_db, beam_bps_per_Hz, bps_per_Hz)
    # result_output(iter, cni_mean, beam_bps, beam_cni_db, beam_bps_per_Hz, bps_per_Hz)
    
    # if printf: print(f"num {num}, iter: {iter}, cni: {np.round(cni_mean, 12)}[dB], {bps_per_Hz}[bps/Hz], {sum(beam_bps):,}[bps], power : {power} band : {bandwidth}")
    if printf: 
        print(f"rad : {sat_radius[0]}m, cni: {np.round(cni_mean, 3)}[dB], {np.round(bps_per_Hz, 3)}[bps/Hz], {sum(beam_bps):,}[bps], carrier:{dBm(np.mean(cni_data[3]))}[dB], interference:{dBm(np.mean(cni_data[4]))}[dB]")

        for beam_num in range(num_of_beam):
            freq_num = determ_freq(freq_beam_list, beam_num)
            print(f"beam_num:{beam_num}, band:{freq_num}, power:{power[beam_num]}, bandwidth:{bandwidth[freq_num]}, cni:{beam_cni_db[beam_num]}, bps/Hz:{beam_bps[beam_num]}, beam:{beam_center[beam_num]}, sat_rad:{sat_radius[beam_num]}, beam_rad:{cni_data[2][beam_num]}, carrier:{dBm(cni_data[3][beam_num])}, interference:{dBm(cni_data[4][beam_num])}")


    return beam_bps


def main(start_end_list) :
    global bps_person_max, power, process_num, output_time, output_path, cni_data
    global sat_radius, dist_from_center, start_sat_radius, end_sat_radius

    process_num         = start_end_list[0]
    start_sat_radius    = start_end_list[1]
    end_sat_radius      = start_end_list[2]
    output_time         = start_end_list[3]
    output_path         = start_end_list[4]

    print(f'プロセス{process_num} 開始 : {setup_time}, {start_sat_radius}から{end_sat_radius}まで')

    for this_sat_radius in range(start_sat_radius, end_sat_radius, 1) :
        sat_radius          = [this_sat_radius/10] * num_of_beam
        dist_from_center    = add_beam(beam_center, sat_radius)
        cni_data            = calc_cni(power, bandwidth)
        
        calc_bitrate(cni_data[0], this_sat_radius, this_sat_radius)

        if cni_plot:
            plot_cni()


    print(f"プロセス{process_num} 終了 : {time.time() - runtime}")
    # plt.show()




power               = list()
freqField           = dict()   # 角周波数ごとにビームを入れる辞書
bps_person_max      = [0] * 7
bps_person_list_num = list()
dl_mw               = mW(DOWNLINK_LOSS)


for iter in range(REPEATED_BEAM):
    if freqField.get(iter) is None:
        freqField[iter] = list()

num_of_beam         = beam_count()[0]
beam_center         = beam_count()[1] #[latitude, longitude]
freq_beam_list      = beam_freq()

power               = initial_power()
bandwidth           = initial_bandwidth()
setup_time = time.time() - runtime

if __name__ == '__main__':

    if output_csv:
        now         = datetime.datetime.now()
        output_time = now.strftime("%m%d-%H%M%S")
        # output_path = "Result\\Result_" + output_time
        output_path = "Result\\Result_ship_and_flight"
        # os.mkdir(output_path)

    start_sat_radius = 10 # 0.1m
    end_sat_radius   = 301 # 0.1m

  
    main([0, circle["sat_radius"][0]*10, circle["sat_radius"][0]*10+1, output_time, output_path])




    # main([0, start_sat_radius, end_sat_radius, output_time, output_path])

    # for this_sat_radius in range(start_sat_radius, end_sat_radius, 10) :
        # sat_radius          = [this_sat_radius/10] * num_of_beam
        # print(sat_radius)
        # dist_from_center    = add_beam(beam_center, sat_radius)
        # dist_from_center_x  = dist_from_center[0]
        # dist_from_center_y  = dist_from_center[1]
        # beam_radius         = calc_cni(power, bandwidth)[2]
        # center_dist_list    = pref_beam_distance()
        # beam_user = np.round(np.array(user_count()))

        # power_list = pd.read_csv("database\\decision_power_8_0927-115200.csv", dtype=np.float16)
        # power_list = power
        # main([power, 0, this_sat_radius, this_sat_radius+1, output_time, output_path])
        # print(f"実行時間 : {time.time() - runtime - setup_time}")

    plt.show()



# print(len(freqField[0][0][0][0]))  # longitude
# print(len(freqField[0][0][0]))     # latitude
# print(len(freqField[0][0]))        # gb, xy
# print(len(freqField[0]))           # beam_num
# print(len(freqField))              # REPEATED_BEAM

# print(f"freqField[{len(freqField)}][{len(freqField[0])}][{len(freqField[0][0])}][{len(freqField[0][0][0])}][{len(freqField[0][0][0][0])}]")
# freqField[REPEATED_BEAM][ビーム数][gb, xy][latitude][longitude]