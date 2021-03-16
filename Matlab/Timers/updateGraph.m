function updateGraph(mTimer,~)
global X T
figure(1)
subplot(2,1,1)
plot(T,X(1,:),'r','linewidth',2)
subplot(2,1,2)
plot(T,X(2,:),'b','linewidth',2)
%refreshdata
drawnow
end

