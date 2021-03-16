"""
Control Interface reader

include all the controllers
"""

import csv
import math
import time
import numpy as np
from control import lqr, dlqr

class Control_Interface_Reader():

    def __init__(self):

        self.cntr_sel = 0
        self.ref = 0
        self.Ts = 0
        self.Kp = 0
        self.tau_i = 0
        self.tau_d = 0
        self.alpha = 0
        self.umax = math.inf
        self.umin = -math.inf

    def update(self):
        data = csv.reader(open('Control_Interface2.csv','r'))
        next(data)
        list = []
        for row in data:
            list.append(row)
        self.cntr_sel = int(list[2][0])
        self.ref = int(list[2][1])
        self.Ts = int(list[5][2])
        func = self.control_sel(self.cntr_sel, self.ref, self.Ts)
        return self.ref, self.Ts, func

    def control_sel(self,argument, ref, Ts):
        switcher = {
            1: self.PID,
            2: self.LQR,
            3: self.MPC
        }
        func = switcher.get(argument, lambda: print("Invalid control option"))
        return func(self.ref,self.Ts)

    def get_lqr_weights(self, nx, num_int):
        R1, R2 =[],[]
        size = nx+num_int
        data = csv.reader(open('Control_Interface2.csv','r'))
        next(data)
        list = []
        for row in data:
            list.append(row)
        for i in range(0,size):
            R1.append(list[i+10][1:4])
        R2 = list[10][4]
        return R1, R2

    def show_table(self):
        data = csv.reader(open('Control_Interface2.csv','r'))
        next(data)
        list = []
        for row in data:
            list.append(row)
        print(list)
    def PID(self, ref, Ts):
        #print("PID")
        #print("Ref: {}".format(self.ref))
        #print("Ts: {}".format(self.Ts))
        return "PID"
    def LQR(self, ref, Ts):
        #print("LQR")
        #print("Ref: {}".format(self.ref))
        #print("Ts: {}".format(self.Ts))
        # LQR_designer()
        return "LQR"
    def MPC(self, ref, Ts):
        #print("MPC")
        #print("Ref: {}".format(self.ref))
        #print("Ts: {}".format(self.Ts))
        # MPC_designer()
        return "MPC"

def main(cntInterface):
    A = np.array([[4, 3], [-4.5, -3.5]])
    B = np.array([[1], [-1]])

    while True:
        R1, R2 = cntInterface.get_lqr_weights(2,1)
        print(np.reshape(R1, (3,3)))
        [K, P, ev] = dlqr(A,B,np.reshape(R1, (3,3)), [R2])
        print(K)
        time.sleep(1)

    
if __name__ == "__main__": 

    cntInterface = Control_Interface_Reader()
    main(cntInterface)
