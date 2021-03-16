function simStop(mTimer,~)
global t x u
str = ['Stopped simulation at time: ',num2str(t),'s. States:'];
disp(str)
disp(x)
delete(mTimer);
end

