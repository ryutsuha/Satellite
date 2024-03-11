import xlrd
import shapely
import openpyxl
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as tick # 目盛り操作に必要なライブラリを読み込みます
import matplotlib.patches as patches

circle = pd.read_csv("result_print.csv")
# circle = pd.read_excel("result_print.xlsx")
# circle = pd.read_excel("beam_mapping.xlsx")
colorlist = {"cl0" : "#888888", "cl1" : "#4FAAD1", "cl2" : "#EBBF00", "cl3" : "#B66427", "cl4" : "#0f4047"}


beam_center = dict()
beam_radius = dict()
beam_freq = dict()

fig, ax = plt.subplots(figsize = (13,13))
df = gpd.read_file('japan.geojson')
# df = gpd.read_file('japan_eez.json')
df.plot(ax = ax, edgecolor='#444', facecolor='white', linewidth = 0.5, aspect="equal")

# plt.scatter(128, 30, marker="+", color="black")
# plt.scatter(146.676333, 30, marker="+", color="black")
# plt.scatter(128, 46.216165, marker="+", color="black")
# plt.scatter(146.676333, 46.216165, marker="+", color="black")

plt.xlim([122, 147])
plt.ylim([20, 47])
plt.gca().xaxis.set_minor_locator(tick.MultipleLocator(1))
plt.gca().yaxis.set_minor_locator(tick.MultipleLocator(1))

for beam_num in range(len(circle)):
    print([circle["longitude"][beam_num], circle["latitude"][beam_num]], circle["beam_radius"][beam_num] / 100, int(circle["band"][beam_num]+1) , colorlist["cl" + str(int(circle["band"][beam_num]+1))])
    # ax.add_patch(patches.Circle(xy=([circle["longitude"][beam_num], circle["latitude"][beam_num]]), radius=circle["beam_radius"][beam_num] / 100, color=colorlist["cl" + str(int(circle["band"][beam_num]+1))],alpha=0.3))
    # ax.text(circle["longitude"][beam_num]-0.2, circle["latitude"][beam_num]-0.2, circle["beam_num"][beam_num])

# plt.savefig("jpmap.png",transparent=True)
plt.show()
