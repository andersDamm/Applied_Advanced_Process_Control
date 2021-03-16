function timerFnc(mTimer, ~)
global data timedrift

data = [data mTimer.UserData];


temp = clock();
X = ['time: ',num2str(temp(end-1)), 'min ',num2str(temp(end)),'s'];
disp(X)
timedrift = [timedrift temp(end)];

end

