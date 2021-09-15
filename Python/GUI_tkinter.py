#---------Imports

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as Tk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


import matplotlib.pyplot as plt
import matplotlib
import OPCUA_SQL as db
import matplotlib.dates as md
import matplotlib.dates as mdates


global y1_arr,y2_arr, u1_arr, u2_arr, r1, r2, t1, t2, t3, t4 
#---------End of imports

conn = db.connect()
rsid = db.getRSID(conn)
print("GUI RSID = {}".format(rsid))

Ts = 0.5 # sampling time [s]
y1_arr, y2_arr, u1_arr, u2_arr, t1, t2, t3, t4 = [], [], [], [], [], [], [], []
fig = plt.figure(figsize = (8,8))
plt.subplots_adjust(wspace=0.8, 
                    hspace=1.5)
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

#x = np.arange(0, 2*np.pi, 0.01)        # x-array

def animate(i, conn, rsid, y1_arr, y2_arr, u1_arr, u2_arr, t1, t2, t3, t4):
    n = 10 # frequency of ticks
    nrows = 100
    # Major ticks every 20, minor ticks every 5
    major_ticks = np.arange(0, 101, 20)
    minor_ticks = np.arange(0, 101, 5)

    # If not Manual Control reference signals should be plottet:
    


    y1 = db.gety1(conn, rsid)
    y2 = db.gety2(conn, rsid)
    
    u1 = db.getu1(conn, rsid)
    u2 = db.getu2(conn, rsid)

    r1 = db.getr1(conn, rsid)
    r2 = db.getr2(conn, rsid)
    cm = db.getControlmode(conn,rsid)[0]

    # Add x and y to lists
    t1.append(y1[1].strftime('%H:%M:%S.%f'))
    y1_arr.append(y1[0])
    t2.append(y2[1].strftime('%H:%M:%S.%f'))
    y2_arr.append(y2[0])

    t3.append(u1[1].strftime('%H:%M:%S.%f'))
    u1_arr.append(u1[0])
    t4.append(u2[1].strftime('%H:%M:%S.%f'))
    u2_arr.append(u2[0])


    # Limit x and y lists to n
    t1 = t1[-nrows:]
    t2 = t2[-nrows:]
    t3 = t3[-nrows:]
    t4 = t4[-nrows:]

    y1_arr = y1_arr[-nrows:]
    y2_arr = y2_arr[-nrows:]
    u1_arr = u1_arr[-nrows:]
    u2_arr = u2_arr[-nrows:]

    ax1.grid(which='minor', alpha=0.2)
    ax1.grid(which='major', alpha=0.5)
    ax2.grid(which='minor', alpha=0.2)
    ax2.grid(which='major', alpha=0.5)
    ax3.grid(which='minor', alpha=0.2)
    ax3.grid(which='major', alpha=0.5)
    ax4.grid(which='minor', alpha=0.2)
    ax4.grid(which='major', alpha=0.5)
    # Draw x and y lists
    ax1.clear()
    ax1.plot(t1, y1_arr, color='blue',label='y1')
    ax2.clear()
    ax2.plot(t2, y2_arr, color='blue',label='y2')
    ax3.clear()
    ax3.step(t3, u1_arr, color='orange',label='u1')
    ax4.clear()
    ax4.step(t4, u2_arr, color='orange',label='u2')

    if(cm>0):
        ax1.axhline(y=r1[0], color='k', linestyle='--',label='r1')
        ax2.axhline(y=r2[0], color='k', linestyle='--',label='r2')

    # Format plot
    plt.subplot(ax1)
    [l.set_visible(False) for (i,l) in enumerate(ax1.xaxis.get_ticklabels()) if i % n != 0]
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title('Tank 1')
    plt.grid()
    plt.ylabel('Height $[\mathrm{cm}]$')
    plt.xlabel('Time (UTC) $[hh:mm:ss.ms]$')
    plt.legend(loc='upper left')
    plt.ylim((0,max(max(y1_arr[:]), r1[0])+5))

    plt.subplot(ax2)
    [l.set_visible(False) for (i,l) in enumerate(ax2.xaxis.get_ticklabels()) if i % n != 0]
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title('Tank 2')
    plt.grid()
    plt.ylabel('Height $[\mathrm{cm}]$')
    plt.xlabel('Time (UTC) $[hh:mm:ss.ms]$')
    plt.legend(loc='upper left')
    plt.ylim((0,max(max(y2_arr[:]), r2[0])+5))

    # Format plot
    plt.subplot(ax3)
    [l.set_visible(False) for (i,l) in enumerate(ax3.xaxis.get_ticklabels()) if i % n != 0]
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title('Pump 1')
    plt.grid()
    plt.ylabel('Flow rate $[\mathrm{cm^3\,s^{-1}}]$')
    plt.xlabel('Time (UTC) $[hh:mm:ss.ms]$')
    plt.legend(loc='upper left')
    plt.ylim((0,max(u1_arr)+5))

    plt.subplot(ax4)
    [l.set_visible(False) for (i,l) in enumerate(ax4.xaxis.get_ticklabels()) if i % n != 0]
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title('Pump 2')
    plt.grid()
    plt.ylabel('Flow rate $[\mathrm{cm^3\,s^{-1}}]$')
    plt.xlabel('Time (UTC) $[hh:mm:ss.ms]$')
    plt.legend(loc='upper left')
    plt.ylim((0,max(u2_arr)+5))

root = Tk.Tk()
root.geometry("1400x1100")

label = Tk.Label(root,text="Four Tank System GUI").grid(column=0, row=0)

lbl1 = Tk.Label(root, text="u1 = ")
lbl1.grid(column=0, row=2)

lbl2 = Tk.Label(root, text="u2 = ")
lbl2.grid(column=0, row=3)

selected = Tk.IntVar()
rad1 = Tk.Radiobutton(root,text='Manual Control', value=0, variable = selected)

rad2 = Tk.Radiobutton(root,text='PID', value=1, variable = selected)

rad3 = Tk.Radiobutton(root,text='LMPC', value=2, variable = selected)

rad4 = Tk.Radiobutton(root,text='NMPC', value=3, variable = selected)

rad1.grid(column=1, row=0)

rad2.grid(column=2, row=0)

rad3.grid(column=3, row=0)

rad4.grid(column=4, row=0)

txt1= Tk.Entry(root,width=10)
txt1.grid(column=1, row=2)
txt2= Tk.Entry(root,width=10)
txt2.grid(column=1, row=3)

lbl3 = Tk.Label(root, text="u1 = {}[cm^3/s] u2 = {}[cm^3/s]".format(db.getu1(conn, rsid)[0], db.getu2(conn, rsid)[0]))
lbl3.grid(column=0, row=4)

def clicked_uurr():
    ts = db.gety1(conn, rsid)
    cm = db.getControlmode(conn, rsid)[0]
    if(selected.get() == 0):
        db.insertu1(conn, rsid, ts[1], txt1.get(), "testu1", cm)
        db.insertu2(conn, rsid, ts[1], txt2.get(), "testu2", cm)
        lbl3.configure(text= "u1 = {}[cm^3/s] u2 = {}[cm^3/s] ".format(db.getu1(conn, rsid)[0], db.getu2(conn, rsid)[0]))
    if (selected.get() > 0):
        db.insertr1(conn, rsid, ts[1], txt1.get(), cm)
        db.insertr2(conn, rsid, ts[1], txt2.get(), cm)
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
   



btn1 = Tk.Button(root, text="Set value for u1 and u2", command=clicked_uurr)
btn1.grid(column=2, row=2)

btn2 = Tk.Button(root, text="Mode select", command=modeEnter)
btn2.grid(column=5, row=0)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().grid(column=0,row=1)


ani = animation.FuncAnimation(fig, animate,fargs=(conn, rsid, y1_arr, \
     y2_arr, u1_arr, u2_arr, t1, t2, t3, t4), interval=Ts*1000)

Tk.mainloop()