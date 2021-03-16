function t = createTimer(Ts,data)

t = timer;
t.UserData = data;
t.StartFcn = @timerStart;
t.TimerFcn = @timerFnc;
t.StopFcn = @timerStop;
t.Period = Ts;
t.StartDelay = 0;
t.ExecutionMode = 'fixedRate';

end