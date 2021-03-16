function t = drawGraph(period)
% A timer for simulating a system

t = timer;
%t.UserData = secondsBreak;
t.TimerFcn = @updateGraph;
t.Period = period;
t.StartDelay = 0
t.ExecutionMode = 'fixedRate';
end 