clear all; close all; clc;
clust = parcluster('local');


%% timer parameter
period1=1;
%period2=0.25;

%% create tasks

obj1= batch(clust,@timerfun_new,1,{period1});
%obj2= batch(clust,@timerfun_new,1,{period2,abstime});


%% submit

load(obj1)

%%
wait(obj2);
r2=fetchOutputs(obj2);
r2{1}



%% clean up
delete(obj1);



