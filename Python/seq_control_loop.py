## Sequential control loop test




import ver4 as VER4
import SQL_Python as sql
import numpy as np
import matplotlib.pyplot as plt
from ver4 import data2collumns

class systemSimulation():
    def __init__(self, x0, A, B, C, Ts):
        self.x = x0
        self.t = 0
        self.y = np.dot(C,self.x)
        self.xdot = np.array([[0.0,0.0]]).T
        self.A = A
        self.B = B
        self.C = C
        self.X = np.array([[0,0]]).T
        self.T = 0
        self.Y = self.y
        self.Ts = Ts


    def forwardEuler(self):
        self.x = self.x + self.xdot*0.1

    def dx_dt(self, u):
        self.xdot = self.A.dot(self.x)+self.B.dot(u)
        self.y    = np.dot(self.C,self.x)
        self.forwardEuler()
        
    def show(self):
        #print(self.x)
        return self.x
def initSystem():
   
    # Init dynimical system
    sys = VER4.initStateSpace()
    # Init control system
    cont = VER4.controlGains()


    return sys, cont


def simulation(u, Ts, N):
    global timeseries, timeseries_u, timetime
    [sys, cont] = initSystem()
    sim1 = systemSimulation(sys.x, sys.A, sys.B, sys.C, Ts)

    timetime = np.linspace(0, Ts*N, N)
    timeseries, timeseries_u = np.array([]), np.array([])
    print("start")
    r = 30
    xi = 0
    Kd = np.array([24.8589443870757,  103.5471783576996])
    Ki =   2.617740102886537
    U = 0
    for i in range(0, N):
        for j in range(0,10):
            sim1.dx_dt(U)
            x = sim1.show()
        e = (r - x[0])*Ts
        xi = xi+e
        U = float(str(-Kd.dot(x)+Ki*xi).lstrip('[').rstrip(']'))
        #U = 10
        timeseries = np.append(timeseries, x[0])
        timeseries_u = np.append(timeseries_u, U)



def main(id = None):
    global timeseries, timeseries_u, timetime
    
    try:
        plt.close('all')
        mainConn = sql.connect()
        results_meas = sql.readAllFromView_meas(mainConn)
        data_meas = data2collumns(results_meas, 3)
        results_input = sql.readAllFromView_input(mainConn)
        data_input = data2collumns(results_input, 2)

            
        Ts = 1

        u = 10
        N = 70

        simulation(u, Ts, N)

        plt.subplot(211)
        plt.title('Ts = 0.01s', fontsize=20)
        axes = plt.gca()
        plt.plot(data_meas[0], data_meas[1],linewidth=3,c = 'b', label='y - Real-time')
        plt.plot(timetime, timeseries,linewidth=3,linestyle = 'dotted', c = 'r', label='y - Sequential')
        plt.legend(loc='lower right')
        plt.xlabel('time [s]', fontsize=14)
        plt.grid()

        plt.subplot(212)
        axes = plt.gca()
        plt.step( data_input[0],  data_input[1],linewidth=3,c = 'b', label='u - Real-time')  # Change such that type is correctly updated.
        plt.step(timetime, timeseries_u,linewidth=3, c = 'r',label='u - Sequential')
        plt.legend(loc='lower right')
        plt.xlabel('time [s]', fontsize=14)
        plt.grid()
        plt.tight_layout()

        plt.show()

        

    except (Exception) as error:
            print(error)
    finally:
            mainConn.close()

if __name__ == '__main__':
    main(99)

