function t = controllerTimer(Ts)
% A timer for simulating a system

t = timer;
%t.UserData = secondsBreak;
%t.StartFcn = @simStart;
t.TimerFcn = @controller;
%t.StopFcn = @simStop;
t.Period = Ts;
t.BusyMode = 'queue';
%t.StartDelay = 0.01
t.ExecutionMode = 'fixedSpacing';
drawnow
end 