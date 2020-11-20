import csv
import glob
import os.path
from os import path
import shutil
import os
import random

def main():
    paths = glob.glob("/Users/joshchung/Desktop/Sampled/*.csv")
    for i in range(86):
        a = random.randint(0, len(paths))
        p = paths[a]
        to = p[:25]+"cross/g9/"+p[33:]
        print(p, to)
        paths.pop(a)
        shutil.move(p,to)


def crossVal():
    pathi = "/Users/joshchung/Desktop/sampleBad/*.csv"
    for fname in glob.glob(pathi):
        with open("/Users/joshchung/PycharmProjects/ArestyResearchGit/Aresty/visual/porosityName.csv", 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([fname[35:]])

if __name__== "__main__":
   main()

#Sampled_t0p4573_x0_y5p808_z0_layer1