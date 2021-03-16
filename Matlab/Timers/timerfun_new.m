function y=timerfun_new(period)
global global_var
t=timer('ExecutionMode','fixedrate','Period',period,'TimerFcn',@(x,y)timercallback);
start(t);
%wait(t);
y=global_var;
function timercallback(x,y)
global global_var

global_var = global_var + 1;
