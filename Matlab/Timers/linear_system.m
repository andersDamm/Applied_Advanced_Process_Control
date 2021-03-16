function xdot = linear_system(t,x,u)
omega = 0.1;
zeta = 0.4;
gain = 0.4213456;

A = [0,       1;
     -omega^2, -2*zeta*omega];
B = [0;omega^2]*gain;

C = [1, 0];
D = 0;
xdot = A*x+B*u;
end

