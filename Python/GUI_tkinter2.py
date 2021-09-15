"""

GUI based on Tkinter and matplotlib animation which is connected to database


"""


#---------Imports

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import Frame,Label,Entry,Button
import tkinter as Tk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


import matplotlib.pyplot as plt
import matplotlib
import OPCUA_SQL as db
import matplotlib.dates as md
import matplotlib.dates as mdates


class GUI(Frame):
    def __init__(self,  master = None):
        Frame.__init__(self, master)
        self.master = master
        self.conn = db.connect()
        self.rsid = db.getRSID(self.conn)
        self.Ts = 0.5 # sampling time [s]
        
        # Init arrays for plotting
        self.y1_arr, self.y2_arr, self.u1_arr, self.u2_arr, self.r1, self.r2, \
         self.t1, self.t2, self.t3, self.t4 = ([] for i in range(10))

        print("GUI RSID = {}".format(self.rsid))

        self.window()


    def window(self):

        self.fig = plt.figure(figsize = (8,8))
        plt.subplots_adjust(wspace=0.8, 
                            hspace=1.5)
        self.ax1 = self.fig.add_subplot(2, 2, 1)
        self.ax2 = self.fig.add_subplot(2, 2, 2)
        self.ax3 = self.fig.add_subplot(2, 2, 3)
        self.ax4 = self.fig.add_subplot(2, 2, 4)

        def clicked_uurr():
            ts = db.gety1(self.conn, self.rsid)
            cm = db.getControlmode(self.conn, self.rsid)[0]
            if(selected.get() == 0):
                db.insertu1(self.conn, self.rsid, ts[1], txt1.get(), "testu1", cm)
                db.insertu2(self.conn, self.rsid, ts[1], txt2.get(), "testu2", cm)
                lbl3.configure(text= "u1 = {}[cm^3/s] u2 = {}[cm^3/s] ".format(db.getu1(conn, rsid)[0], db.getu2(conn, rsid)[0]))
            if (selected.get() > 0):
                db.insertr1(self.conn, self.rsid, ts[1], txt1.get(), cm)
                db.insertr2(self.conn, self.rsid, ts[1], txt2.get(), cm)
                lbl3.configure(text= "r1 = {}[cm] r2 = {}[cm] ".format(db.getr1(conn, rsid)[0], db.getr2(conn, rsid)[0]))

        def modeEnter():
            ts = db.gety1(conn, rsid)
            db.insertControlmode(conn, rsid, ts[1],selected.get())
            if (selected.get() == 0):
                lbl1.configure(text = "u1 = ")
                lbl2.configure(text = "u2 = ")
                lbl3.configure(text= "u1 = {}[cm^3/s] u2 = {}[cm^3/s] ".format(db.getu1(conn, rsid)[0], db.getu2(conn, rsid)[0]))
            if (selected.get() > 0):
                lbl1.configure(text = "r1 = ")
                lbl2.configure(text = "r2 = ")
                lbl3.configure(text= "r1 = {}[cm] r2 = {}[cm] ".format(db.getr1(conn, rsid)[0], db.getr2(conn, rsid)[0]))

        def animate(i):
            n = 10 # frequency of ticks
            nrows = 100 # Max number of data points on plots

            # Major ticks every 20, minor ticks every 5 (to make plots more nice)
            major_ticks = np.arange(0, 101, 20)
            minor_ticks = np.arange(0, 101, 5)

            y1 = db.gety1(self.conn, self.rsid)
            y2 = db.gety2(self.conn, self.rsid)
            
            u1 = db.getu1(self.conn, self.rsid)
            u2 = db.getu2(self.conn, self.rsid)

            r1 = db.getr1(self.conn, self.rsid)
            r2 = db.getr2(self.conn, self.rsid)
            cm = db.getControlmode(self.conn, self.rsid)[0]

            # Add x and y to lists
            self.t1.append(y1[1].strftime('%H:%M:%S.%f'))
            self.y1_arr.append(y1[0])
            self.t2.append(y2[1].strftime('%H:%M:%S.%f'))
            self.y2_arr.append(y2[0])

            self.t3.append(u1[1].strftime('%H:%M:%S.%f'))
            self.u1_arr.append(u1[0])
            self.t4.append(u2[1].strftime('%H:%M:%S.%f'))
            self.u2_arr.append(u2[0])


            # Limit arrays to maximum number of points
            self.t1 = self.t1[-nrows:]
            self.t2 = self.t2[-nrows:]
            self.t3 = self.t3[-nrows:]
            self.t4 = self.t4[-nrows:]

            self.y1_arr = self.y1_arr[-nrows:]
            self.y2_arr = self.y2_arr[-nrows:]
            self.u1_arr = self.u1_arr[-nrows:]
            self.u2_arr = self.u2_arr[-nrows:]

            self.ax1.grid(which='minor', alpha=0.2)
            self.ax1.grid(which='major', alpha=0.5)
            self.ax2.grid(which='minor', alpha=0.2)
            self.ax2.grid(which='major', alpha=0.5)
            self.ax3.grid(which='minor', alpha=0.2)
            self.ax3.grid(which='major', alpha=0.5)
            self.ax4.grid(which='minor', alpha=0.2)
            self.ax4.grid(which='major', alpha=0.5)
            # Draw x and y lists
            self.ax1.clear()
            self.ax1.plot(self.t1, self.y1_arr, color='blue',label='y1')
            self.ax2.clear()
            self.ax2.plot(self.t2, self.y2_arr, color='blue',label='y2')
            self.ax3.clear()
            self.ax3.step(self.t3, self.u1_arr, color='orange',label='u1')
            self.ax4.clear()
            self.ax4.step(self.t4, self.u2_arr, color='orange',label='u2')

            if(cm>0):
                self.ax1.axhline(y=r1[0], color='k', linestyle='--',label='r1')
                self.ax2.axhline(y=r2[0], color='k', linestyle='--',label='r2')

            # Format plot
            plt.subplot(ax1)
            [l.set_visible(False) for (i,l) in enumerate(self.ax1.xaxis.get_ticklabels()) if i % n != 0]
            plt.xticks(rotation=45, ha='right')
            plt.subplots_adjust(bottom=0.30)
            plt.title('Tank 1')
            plt.grid()
            plt.ylabel('Height $[\mathrm{cm}]$')
            plt.xlabel('Time (UTC) $[hh:mm:ss.ms]$')
            plt.legend(loc='upper left')
            plt.ylim((0,max(max(self.y1_arr[:]), self.r1[0])+5))

            plt.subplot(ax2)
            [l.set_visible(False) for (i,l) in enumerate(self.ax2.xaxis.get_ticklabels()) if i % n != 0]
            plt.xticks(rotation=45, ha='right')
            self.plt.subplots_adjust(bottom=0.30)
            plt.title('Tank 2')
            plt.grid()
            plt.ylabel('Height $[\mathrm{cm}]$')
            plt.xlabel('Time (UTC) $[hh:mm:ss.ms]$')
            plt.legend(loc='upper left')
            plt.ylim((0,max(max(self.y2_arr[:]), self.r2[0])+5))

            # Format plot
            self.plt.subplot(ax3)
            [l.set_visible(False) for (i,l) in enumerate(self.ax3.xaxis.get_ticklabels()) if i % n != 0]
            plt.xticks(rotation=45, ha='right')
            plt.subplots_adjust(bottom=0.30)
            plt.title('Pump 1')
            plt.grid()
            plt.ylabel('Flow rate $[\mathrm{cm^3\,s^{-1}}]$')
            plt.xlabel('Time (UTC) $[hh:mm:ss.ms]$')
            plt.legend(loc='upper left')
            plt.ylim((0,max(self.u1_arr)+5))

            plt.subplot(ax4)
            [l.set_visible(False) for (i,l) in enumerate(self.ax4.xaxis.get_ticklabels()) if i % n != 0]
            plt.xticks(rotation=45, ha='right')
            plt.subplots_adjust(bottom=0.30)
            plt.title('Pump 2')
            plt.grid()
            plt.ylabel('Flow rate $[\mathrm{cm^3\,s^{-1}}]$')
            plt.xlabel('Time (UTC) $[hh:mm:ss.ms]$')
            plt.legend(loc='upper left')
            plt.ylim((0,max(self.u2_arr)+5))


        # ---- Labels and buttons -----

        label = Tk.Label(self,text="Four Tank System GUI").grid(column=0, row=0)

        lbl1 = Tk.Label(self, text="u1 = ")
        lbl1.grid(column=0, row=2)

        lbl2 = Tk.Label(self, text="u2 = ")
        lbl2.grid(column=0, row=3)

        self.selected = Tk.IntVar()
        rad1 = Tk.Radiobutton(self,text='Manual Control', value=0, variable = self.selected)

        rad2 = Tk.Radiobutton(self,text='PID', value=1, variable = self.selected)

        rad3 = Tk.Radiobutton(self,text='LMPC', value=2, variable = self.selected)

        rad4 = Tk.Radiobutton(self,text='NMPC', value=3, variable = self.selected)

        rad1.grid(column=1, row=0)

        rad2.grid(column=2, row=0)

        rad3.grid(column=3, row=0)

        rad4.grid(column=4, row=0)

        txt1= Tk.Entry(self,width=10)
        txt1.grid(column=1, row=2)
        txt2= Tk.Entry(self,width=10)
        txt2.grid(column=1, row=3)

        lbl3 = Tk.Label(self, text="u1 = {}[cm^3/s] u2 = {}[cm^3/s]".format(db.getu1(self.conn, self.rsid)[0], \
            db.getu2(self.conn, self.rsid)[0]))
        lbl3.grid(column=0, row=4)


        btn1 = Tk.Button(self, text="Set value for u1 and u2", command=clicked_uurr)
        btn1.grid(column=2, row=2)

        btn2 = Tk.Button(self, text="Mode select", command=modeEnter)
        btn2.grid(column=5, row=0)


        canvas = FigureCanvasTkAgg(self.fig, master=self)
        canvas.get_tk_widget().grid(column=0,row=1)
        print("herher")
        self.ani = animation.FuncAnimation(self.fig, animate, interval=self.Ts*1000)
        print("herher")




if __name__ == "__main__":

    root = Tk.Tk()
    root.geometry("1400x1100")
    gui = GUI(root)
    print("her")
    Tk.mainloop()