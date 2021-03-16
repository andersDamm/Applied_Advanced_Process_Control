%% MUTEX test

clear all; clc; close all;


global data delay
data = []
delay = 5;


timer1 = createTimer(1,'.')
timer2 = createTimerWithDelay(10,'+')
start(timer2)
start(timer1)

%% 
stop(timer1)
stop(timer2)