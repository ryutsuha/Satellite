import shapely
import geopandas as gpd
import matplotlib.pyplot as plt

# eez = gpd.read_file('japan.geojson')
eez = gpd.read_file('japan_eez.json')
hoppou = gpd.read_file('japan_hoppou.json')
dev_zone = gpd.read_file('japan_korea_dev_zone.json')
senkaku = gpd.read_file('japan_senkaku.json')


poly = {'lon' : 138.18111, 'lat' : 36.65139}
poly = {'lon' : 132, 'lat' : 31.6}

poly0 = eez.iloc[0].geometry
p_nagano = shapely.geometry.point.Point(poly['lon'], poly['lat']) # 県庁所在地
print(poly['lon'], poly['lat'], poly0.contains(p_nagano)) # > True

fig, ax = plt.subplots(figsize = (10,10))
eez.plot(ax = ax, edgecolor='#444', facecolor='white', linewidth = 0.5, aspect="equal")
hoppou.plot(ax = ax, edgecolor='#444', facecolor='white', linewidth = 0.5, aspect="equal")
dev_zone.plot(ax = ax, edgecolor='#444', facecolor='white', linewidth = 0.5, aspect="equal")
senkaku.plot(ax = ax, edgecolor='#444', facecolor='white', linewidth = 0.5, aspect="equal")


plt.scatter(poly['lon'], poly['lat'], marker="+", color="black")
plt.show()
