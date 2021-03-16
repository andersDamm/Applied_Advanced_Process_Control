%% ---- Timers in MATLAB ----

%% Init
clear all; close all; clc;
format long
%% Global Variables
global data timedrift delay
data = []; timedrift = [];
delay = 1000 % global variable used for three nested forloops
%% Create timer
timer = createTimer(0.1,'.');
timer2 = createTimerWithDelay(2,'+');
%% Start Timer
start(timer)
start(timer2)
%% Stop Timer
stop(timer)
stop(timer2)
%% Plot time differences
time_diff = abs(timedrift(1:end-1)-timedrift(2:end)); x = linspace(1,timedrift(end),length(time_diff));
figure(1)
plot(x,time_diff,'b','linewidth',2)
hold on
yline(0.1,'r--','linewidth',3)
ylim([0.0 1.5])
xlim([1 floor(timedrift(end))])
legend('Real time difference','Ideal time difference','interpreter','latex','location','northwest')
xlabel({'time [s]'},'fontsize',14,'interpreter','latex')
ylabel({'$\Delta$ time [s]'},'fontsize',14,'interpreter','latex')
grid on