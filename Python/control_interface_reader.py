"""
Control Interface reader

include all the controllers
"""

import csv
import math
import time
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
        data = csv.reader(open('Control_Interface.csv','r'))
        next(data)
        list = []
        for row in data:
            list.append(row)
        self.cntr_sel = int(list[2][0])
        self.ref = int(list[2][1])
        self.Ts = float(list[5][2])
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


    
if __name__ == "__main__": 

    cntInterface = Control_Interface_Reader()
    cntInterface.update()
