%% System used
clear all; close all; clc;

zeta = 0.35;
omega_n = 0.070;
gain = 0.678;

A = [0,1;
    -omega_n^2, -omega_n*zeta*2];
B = [0;omega_n^2]*gain
C = [1,0]

sys_ss = ss(A,B,C,0)

step(sys_ss)

%% Controller
close all
Ts = 1;
[F,G] = c2d(A,B,Ts)

eig(A)
1/Ts*log(eig(F))

sys_ss_d = ss(F,G,C,0,Ts)

figure(1)
step(sys_ss)
hold on
step(sys_ss_d)
legend('Continuous-time','Discrete-time')

Fa = [F zeros(2,1)
      -C*Ts 1]
Ga = [G;0]

R1 = diag([1 0 100])
R2 = 1000;
K_lqr = dlqr(Fa,Ga, R1,R2)

Ki = -K_lqr(end)
Kd = K_lqr(1:end-1)


sys_cl = ss(Fa-Ga*K_lqr,[0,0,Ts]',[1,0,0], 0, Ts)



figure(1)
step(sys_cl)
legend('Continuous-time','Discrete-time','closed loop')

