%% Gaussian Functions Aproximation using Self Oorganizing Map(SOM) and Radial Basis Function(RBF)
%  Saeid_Moradi -> saeid.moradi436@gmail.com

%% Create Functions:

x = -5:0.1:5;
y = -5:0.1:5;

q1 = 1;        % Sigma For The First Gaussian Function
q2 = 1;        % Sigma For The Second Gaussian Function
q3 = .7;       % Sigma For The Third Gaussian Function
q4 = .7;       % Sigma For The Forth Gaussian Function

u1 = [2;3];    % Center For The First Gaussian Function
u2 = [1,-1];   % Center For The Second Gaussian Function
u3 = [-3;-2];  % Center For The Third Gaussian Function
u4 = [-2,1];   % Center For The Forth Gaussian Function

[X,Y] = meshgrid(x,y);

Func1 = (1/(q1.*sqrt(2.*pi))).*exp(-((X-u1(1)).^2+(Y-u1(2)).^2)/(2*q1.^2)); % First Gaussian Function
Func2 = (1/(q2.*sqrt(2.*pi))).*exp(-((X-u2(1)).^2+(Y-u2(2)).^2)/(2*q2.^2)); % Second Gaussian Function
Func3 = (1/(q3.*sqrt(2.*pi))).*exp(-((X-u3(1)).^2+(Y-u3(2)).^2)/(2*q3.^2)); % Third Gaussian Function
Func4 = (1/(q4.*sqrt(2.*pi))).*exp(-((X-u4(1)).^2+(Y-u4(2)).^2)/(2*q4.^2)); % Forth Gaussian Function

SumFunc = Func1 + Func2 + Func3 + Func4; % Sum Of Gaussian Functions


figure(1);
surf(X,Y,SumFunc);
title('Main Function')

%% Sampling The Gaussian Functions:

Sample1 = [u1(1) + q1*rand(1,30); u1(2) + q1*rand(1,30)]; %Sampling First Function
Sample2 = [u2(1) + q2*rand(1,30); u2(2) + q2*rand(1,30)]; %Sampling Second Function
Sample3 = [u3(1) + q3*rand(1,30); u3(2) + q3*rand(1,30)]; %Sampling Third Function
Sample4 = [u4(1) + q4*rand(1,30); u4(2) + q4*rand(1,30)]; %Sampling Fourth Function

Samples = [Sample1 Sample2 Sample3 Sample4];   % Sampling Data

XSampling = Samples(1,:);
YSampling = Samples(2,:);

F1 = (1/(q1.*sqrt(2.*pi))).*exp(-((XSampling-u1(1)).^2 + (YSampling-u1(2)).^2)/(2*q1.^2));
F2 = (1/(q2.*sqrt(2.*pi))).*exp(-((XSampling-u2(1)).^2 + (YSampling-u2(2)).^2)/(2*q2.^2));
F3 = (1/(q3.*sqrt(2.*pi))).*exp(-((XSampling-u3(1)).^2 + (YSampling-u3(2)).^2)/(2*q3.^2));
F4 = (1/(q4.*sqrt(2.*pi))).*exp(-((XSampling-u4(1)).^2 + (YSampling-u4(2)).^2)/(2*q4.^2));

Target = F1 + F2 + F3 + F4;   % Targets


figure(2);
plot(Samples(1,:),Samples(2,:),'.');
title('Sampled Points')

%% Find Centers With SOM
H = 4; % Number Of RBF Neurons
Centers = zeros(2,H);

NetSom = selforgmap([H,1],100);
NetSom = train(NetSom,Samples);
Centers = NetSom.iw{1,1}';

%% Find Nearest Center For Each Case of 120 Cases

Distance = zeros(120,H);
for i = 1 : H
 
    Diffs = bsxfun(@minus, Samples, Centers(:,i)); % Subtract Centroid i From All Data
    SqrdDiffs = Diffs .^ 2;                        % Square The Differences
    Distance(:, i) = sum(SqrdDiffs, 1);            % Take The Sum Of The Squared Differences

end

[MinVals,Memberships] = min(Distance, [], 2);

%% Find BETA = 1./(2.*sigma.^2)

Sigma = zeros(1,H);

 for i = 1:H
     
     Center1 = Centers(:,i);                   % Select The Cluster Center
     Members = Samples(:,(Memberships' == i)); % Select All Of The Members Of This Cluster
     
     Differences = bsxfun(@minus, Members, Center1); % Subtract The Center Vector From Each Of The Member Vectors
     SqrdDifferences = sum(Differences .^ 2, 2);     % Take The Sum Of The Squared Differences
     Eucdistances = sqrt(SqrdDifferences);           % Take The Square Root To Get The Euclidean Distance
     Sigma(1,i) = mean(Eucdistances) + .01;          % Compute The Tverage Euclidean Distance And Use This As Sigma
 
 end
    
    % Compute The Beta Values From The Sigma
    Beta = 1./(2.*Sigma.^2);
    
%% Find Activated Values Of Hidden Neurons.

ActiveData = zeros(120,H);

for i = 1 : 120
    
    Input = Samples(:,i);
    Diffs = bsxfun(@minus, Centers, Input); % Get the activation for all RBF neurons for this input.
    SqrdDists = sum(Diffs .^ 2, 1);
    ActiveData(i, :) = exp(-Beta .* SqrdDists);

end

ActiveData = [ones(120, 1), ActiveData]; % Add Bias

%% Find Weights From Hidden To OutPut (Thetas)

Theta(:,1) = pinv(ActiveData' * ActiveData) * ActiveData' * Target';
        
%% Find MSE 

OutPut = zeros(1,120);

for i = 1:120
    
    Input = Samples(:,i);         
    Diffs = bsxfun(@minus, Centers, Input);
    SqrdDists = sum(Diffs .^ 2, 1);
    
    Phis = [1 exp(-Beta .* SqrdDists)]; % Calculate Activated Values And Add A Bias.
    OutPut(1,i) = Phis*Theta;           % Apply The Weights

end

Ejj= ( Target-OutPut) .^ 2 ;
rmSE=sqrt(mean(mean(Ejj)));  % Total RMSE
display(['Total RMSE = ',num2str(rmSE)]);
        
%% Estimation Of The Finction Using Found Centers, Betas And Thetas

xff=-5:0.1:5;
yff=-5:0.1:5;
sIze1=size(xff);
sIze2=size(yff);       
Estimated=zeros(sIze1(2),sIze2(2));
[XFF,YFF]=meshgrid(xff,yff);

for i=1:sIze1(2);
    for j=1:sIze2(2);
        
        Input=[XFF(i,j);YFF(i,j)];
        Diffs = bsxfun(@minus, Centers, Input);
        SqrdDists = sum(Diffs .^ 2, 1);
        Phis = [1 exp(-Beta .* SqrdDists)]; % Calculate Activated Values And Add A Bias.
        Estimated(i,j) = Phis*Theta;        %Apply The Weights.
    
    end
end

figure(3);
surf(XFF,YFF,Estimated);
title('Estimated Function')