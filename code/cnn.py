# feasture extraction
import numpy as np
from numpy import random
import csv
from os import path
import glob

def main():
    print("here")
    filtering()
def filtering():
    filter = np.zeros((3,3))
    sharpen = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    print(sharpen)
    return
if __name__ == "__main__":
    main()