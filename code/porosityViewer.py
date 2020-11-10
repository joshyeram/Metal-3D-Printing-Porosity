import csv
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

def main():
    plt.figure(figsize=(25, 25))
    ax = plt.axes(projection="3d")
    name = "/Users/joshchung/Downloads/PorosityIndexData.csv"
    yBad = np.zeros(71)
    zBad = np.zeros(71)
    xBad = np.zeros(71)
    yGood = np.zeros(1493)
    zGood = np.zeros(1493)
    xGood = np.zeros(1493)
    b = 0
    g = 0
    with open(name, newline='') as csvfile:
        data_list = list(csv.reader(csvfile))
        for i in range(1,len(data_list)):
            if(int(data_list[i][10])==1):
                yBad[b] = float(data_list[i][4]+"."+data_list[i][5])
                zBad[b] = float(data_list[i][6]+"."+data_list[i][7])
                b+=1
            else:
                yGood[g] = float(data_list[i][4]+"."+data_list[i][5])
                zGood[g] = float(data_list[i][6]+"."+data_list[i][7])
                g+=1


    ax.scatter3D(xGood, yGood, zGood, color="green")
    ax.scatter3D(xBad, yBad, zBad, color="red")
    print(yGood)
    plt.show()

if __name__ == "__main__":
    main()
