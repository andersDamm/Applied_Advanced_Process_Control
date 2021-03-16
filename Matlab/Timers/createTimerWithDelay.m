function t = createTimerWithDelay(Ts,data)


t = timer;
t.UserData = data;
t.StartFcn = @timerStart;
t.TimerFcn = @timerfnc_w_delay;
t.StopFcn = @timerStop;
t.Period = Ts;
t.StartDelay = 0
t.ExecutionMode = 'fixedRate';

end