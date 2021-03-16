function timerfnc_w_delay(mTimer, ~)
global data delay

data = [data mTimer.UserData];
sum = 0;
for i = 1:delay
    for j =1:delay
        for k = 1:delay
            sum = sum +1;
        end
    end
end
end