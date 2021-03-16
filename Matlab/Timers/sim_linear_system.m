function sim_linear_system(mTimer,~)
global t u x X T u_lqr
%[~,Xk] = ode15s(@linear_system,[t t+1],x,[],u);
[~,Xk] = ode15s(@linear_system,[t t+1],x,[],u_lqr);

% Euler integration
%x_dot = linear_system(t,x,u);
%x = x+0.1*x_dot;
x = Xk(end,:)';
X = [X x];
T = [T t];
t = t+0.01;
%disp(x)
end

