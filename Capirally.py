import sys
import netCDF4
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import os

# Safety-science.netcdf_cap
# Date: 2018/02/24
# Filename: netcdf_cap

__author__ = 'tetra-r124'
__date__ = "2018/02/24"


class netcdf_cap(object):
    def __init__(self):
        self.__ROOT = os.path.dirname(os.path.abspath(sys.argv[0]))
        self.__EXE_PATH = sys.executable
        self.__ENV_PATH = os.path.dirname(self.__EXE_PATH)
        self.__LOG = os.path.join(self.__ENV_PATH, 'log')

    def main(self):
        os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))
        os.chdir('.\\Capirally')
        nc = netCDF4.Dataset('010_oligo measure-LIF - Channel 1.cdf')
        print(nc)
        result = nc.variables['actual_sampling_interval']
        print(result)
        return

    def asc(self, subplot=True, labels=True):
        os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))
        path = '.\\Capirally\\print'
        n = len(os.listdir(path)) - 1
        os.chdir(path)
        for i, file in enumerate(glob.glob('*.asc')):
            print(file)
            label = file.replace('.asc', '')
            print(label)
            if i == 0:
                df = pd.read_csv(file, skiprows=6, delimiter='\t', nrows=5)
                Hz, data = int(float(df.iloc[0, 0])), int(df.iloc[1, 0])
                time = np.arange(1 / Hz, data * 1 / Hz + 1 / Hz, 1 / Hz) / 60
            df2 = pd.read_csv(file, skiprows=12)
            df3 = df2[0:data]
            if subplot == True:
                plt.subplot(n, 1, i + 1)
            plt.xlim(6, 12)
            plt.xlabel("min")
            plt.ylabel("fluorescence intensity")
            plt.plot(time, df3, label=label)
            if labels == True:
                plt.legend(loc='upper right')
            if i == n - 1:
                break
        plt.show()


if __name__ == '__main__':
    netcdf_cap = netcdf_cap()
    netcdf_cap.asc(subplot=True, labels=False)
    # netcdf_cap.asc(subplot=False,labels=False)