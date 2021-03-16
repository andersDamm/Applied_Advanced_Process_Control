"""
 ------ Ver 4 ------
Real-time simulation of simple 2nd order system + postgreSQL

date: 15/03/2021
@author: Anders Hilmar Damm Andersen

"""

# Libraries
import time
import matplotlib.pyplot as plt
import numpy as np
import threading
import csv
from matplotlib.animation import FuncAnimation

# My packages/services
from control_interface_reader import Control_Interface_Reader
import SQL_Python as sql


class systemTimer():
    def __init__(self, x0, A, B, C, Ts, conn, id):
        self.x = x0
        self.t = time.time()
        self.y = np.dot(C,self.x)
        self.xdot = np.array([[0.0,0.0]]).T
        self.A = A
        self.B = B
        self.C = C
        self.X = np.array([[0,0]]).T
        self.T = time.time()
        self.Y = self.y
        self.Ts = Ts
        self.conn = conn
        self.id = id
        self.time_df = 0.0
        self.time_int = 0.0
        self.t = 0.0

    def forwardEuler(self):

        self.x = self.x + self.xdot*(self.t-self.time_int)
        print("Time internal:", self.t-self.time_int)

    def dx_dt(self):
        global startTime

        u = sql.get_u(self.conn, self.id)
        self.xdot = self.A.dot(self.x)+self.B.dot(u)
        self.t = time.time()-startTime
        self.forwardEuler()
        #print("u = {}. time = {}".format(u,t))
        sql.insert_meas(self.conn, self.id, self.t, str(self.x[0]).lstrip('[').rstrip(']'), str(self.x[1]).lstrip('[').rstrip(']')) # str(states[0]).lstrip('[').rstrip(']')
        #print("time",t)
        self.time_int = self.t
        
    def timerFnc(self):
        while True:
            self.dx_dt()
            self.time_df = (self.Ts-time.time() % self.Ts)
            time.sleep(self.time_df)
            


class controller():

    def __init__(self,Kd, Ki, Ts, conn, id):
        self.Kd = Kd
        self.Ki = Ki
        self.xi = 0
        self.Ts = Ts
        self.conn = conn
        self.id = id
        self.u_old = 0
        self.time_df = 0.0
        self.t = 0.0
        self.t_internal = 0.0

    def control(self, r):
        global startTime
        (x1, x2) = sql.get_states(self.conn, self.id)
        self.t = sql.get_time_from_meas(self.conn, self.id)
        e = (r - x1)*self.Ts
        self.xi = self.xi+e
        x = np.array([[x1,x2]]).T
        u = -self.Kd.dot(x)+self.Ki*self.xi
        #t = time.time()-startTime
        sql.insert_input(self.conn, self.id, self.t, x1, x2, str(u).lstrip('[').rstrip(']'))

    def timerFnc(self, cntInterface):
        while True:
            r, self.Ts, _ =  cntInterface.update()
            
            self.control(r)
            self.time_df = self.Ts-time.time() % self.Ts
            time.sleep(self.time_df)
def animate(i, cntInterface, conn, id):
    try:
        r, Ts, ctr_type =  cntInterface.update()
        meas = sql.read_meas(conn, id, 100)
        data_meas = data2collumns(meas, 3)
        inp = sql.read_input(conn, id, 100)
        data_input = data2collumns(inp, 2)
        plt.clf()

        plt.subplot(211)
        axes = plt.gca()
        plt.axhline(y=r, color='k', linestyle='--',label='r')
        plt.plot(data_meas[0], data_meas[1],c = 'b', label='y')
        plt.legend(loc='upper left')
        plt.xlabel('time [s]', fontsize=14)

        plt.subplot(212)
        axes = plt.gca()
        plt.step(data_input[0], data_input[1],c = 'r', label='u (Type: {}, Ts = {})'.format(ctr_type,Ts))  # Change such that type is correctly updated.
        plt.legend(loc='upper left')
        plt.xlabel('time [s]', fontsize=14)

        plt.tight_layout()
    except (Exception) as error:
        print(error)
        print("An exception occurred in the animation")
        # Can I do something smart here?

def plotter(cntInterface, conn, id):
    plt.style.use('fivethirtyeight')
    time.sleep(2)
    ani = FuncAnimation(plt.gcf(), animate, fargs=(cntInterface, conn, id), interval=0.1*1000)
    plt.tight_layout()
    plt.show()

def data2collumns(data, num_col):
    num_row = len(data)
    col = [[0 for i in range(num_row)] for j in range(num_col)]
    for i in range(num_row):
        for j in range(num_col):
            col[j][i] = data[i][j]
    return col
            
class initStateSpace():

    def __init__(self):
        self.x = np.array([[0,0]]).T
        self.A = np.array([[0.0, 1.0],[-0.0049,  -0.0490]])
        self.B = np.array([[0.0, 0.003322]]).T
        self.C = np.array([1.0, 0.0])
        print(self.A)

    def x(self):
        return self.x

    def A(self):
        return self.A

    def B(self):
        return self.B

    def C(self):
        return self.C

class controlGains():
    def __init__(self):
        self.Kd = np.array([ 24.8589443870757,  103.5471783576996])
        self.Ki =   2.617740102886537
    def Kd(self):
        return self.Kd
    def Ki(self):
        return self.Ki

def initSystem(conn):
    global startTime
    startTime = time.time()
    id = sql.create_session(conn)

    # Init dynimical system
    sys = initStateSpace()
    # Init control system
    cont = controlGains()

    cntInterface = Control_Interface_Reader()

    return sys, cont, cntInterface, id


def main():

    # Sampling times
    Ts_sim        = 0.1 # Sampling time of system/plant 
    Ts_controller = 2.0 # Sampling time of controller   
    Ts_GUI        = 0.1 # Sampling time plotter/GUI

    try:
        mainConn = sql.connect()
        simConn = sql.connect()
        contConn = sql.connect()
        [sys, cont, cntInterface, id] = initSystem(mainConn)

        sim1 = systemTimer(sys.x, sys.A, sys.B, sys.C, Ts_sim, simConn, id)
        cont1 = controller(cont.Kd, cont.Ki, Ts_controller,contConn, id)

        sim_thread = threading.Thread(target=sim1.timerFnc)
        control_thread = threading.Thread(target=cont1.timerFnc, args = (cntInterface,))
        sim_thread.start()

        time.sleep(1)

        #print('Run controller now')
        control_thread.start()
        plotter(cntInterface, mainConn, id)
        
        
    except (Exception) as error:
        print(error)
    finally:
        mainConn.close()
        simConn.close()
        contConn.close()


if __name__ == '__main__':
    main()
