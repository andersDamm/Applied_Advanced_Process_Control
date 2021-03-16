%% ---- Timers in MATLAB ----

%% Init
clear all; close all; clc;
format long
%% Global Variables
global data timedrift
data = []; timedrift = [];
%% Create timer
timer = createTimer(0.1,'.');
%% Start Timer
start(timer)
%% Stop Timer
stop(timer)
%% Plot time differences
time_diff = abs(timedrift(1:end-1)-timedrift(2:end)); x = linspace(1,timedrift(end),length(time_diff));
figure(1)
plot(x,time_diff,'b','linewidth',2)
ylim([0.05 0.15])
xlim([1 floor(timedrift(end))])
xlabel({'time [s]'},'fontsize',14,'interpreter','latex')
ylabel({'$\Delta$ time [s]'},'fontsize',14,'interpreter','latex')
grid on