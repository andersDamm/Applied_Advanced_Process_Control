function t = simulateTimer(Ts)
% A timer for simulating a system

t = timer;
t.UserData = Ts;
t.StartFcn = @simStart;
t.TimerFcn = @sim_linear_system;
t.StopFcn = @simStop;
t.Period = Ts;
t.StartDelay = 0
t.ExecutionMode = 'fixedRate';
end 