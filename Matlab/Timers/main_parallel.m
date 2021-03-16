%% Making the time delay program run in parallel

clear all; close all; clc;


clust = parcluster('local');

global data delay
data = []
delay = 1500;


timer1 = createTimer(1,'.');
timer2 = createTimerWithDelay(10,'+');
%% Start timer
disp('Starting timers')
start(timer1)
start(timer2)
%% stop timer
stop(timer1)
stop(timer2)
