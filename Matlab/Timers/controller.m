function controller(mTimer,~)
global x r Kd Ki xi Ts_controller u_lqr t t_past
C = [1, 0];

e = (r-C*x)*Ts_controller;
xi = xi+e;
u_lqr = -Kd*x+Ki*xi;
disp('u_lqr')
disp(u_lqr);
disp('t_past')
disp(t_past);
disp('t')
disp(t);
disp('Difference')
disp(t-t_past);
t_past = t
%u_lqr = -Kd*x + r

end

