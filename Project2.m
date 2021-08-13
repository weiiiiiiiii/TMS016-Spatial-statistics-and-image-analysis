% For reproducibility
rng('default')  
permeability=load('permeability.mat');
%whos -file permeability.mat;
%  Name       Size              Bytes  Class     Attributes
%  Y         60x220            105600  double    
permeability=permeability.Y;
figure(1)
imagesc(permeability)
Istacked = permeability(:);
% Cross varidation (train: 70%, test: 30%, validation:0%)
%[trainA,valA,testA] = dividerand (Y.', .7, 0,.3);
%[trainA,valA,testA] = dividerand (Istacked', .7, 0,.3);
%size of train:    42   220
%size of test:    18   220
%%
% Separate to training and test data
%trainData = trainA.'; 
%testData = testA.';

%Generate the "measured" data set by adding independent mean-zero Gaussian noise, with standard deviation ?
%A_wnoise = Y + sqrt(variance)*randn(size(Y)) + meanValue;
noise = Istacked + 1*randn(size(Istacked)); %standard deviation=1
%noise = Y + 3*randn(size(Y)); %standard deviation=3

%%
% the K-means algorithm
%   [idx,pars]=normmix_kmeans(x,K,maxiter,verbose) uses the K-means algorithm to
%   estimate a Gaussian mixture model.
[idx,pars]=normmix_kmeans(Istacked,2,10,2);
%choose k=2

% [cl,p]=normmix_classify(x,pars) classifies an image based on a
% Gaussian mixture model estimated using normmix_sgd.
[cl,p]=normmix_classify(Istacked,pars);
Istacked=double(Istacked)/255; 
%Istack = reshape(Y,[size(Y,1)*size(Y,2) size(Y,3)]); 
for k=1:4 
    I_class = Istacked; 
    I_class(cl~=k,:)=256; 
    figure(2)
    subplot(2,2,k)
    imagesc(reshape(I_class,[60 220])); % image is 60*220
    title(['k=',num2str(k)])
    axis image; 
end

I_class2 = Istacked; 
I_class2(cl~=2,:)=256; 
%% Comparing the original image to the segmented image using k-means with k=2
figure(3)
subplot(1, 2, 1), imagesc(permeability)
title('Original image')
axis image;
subplot(1, 2, 2), imagesc(reshape(I_class2,[60 220]));
title('Segmentation using k-means with k=2')
axis image;
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Gaussian mixture model
x=Istacked; % n-by-d matrix (column-stacked image with n pixels)
K=2;% number of classes
Niter=20; %Number of iterations to run the algorithm
step0=0.0001; %The initial step size
plotflag=1;%$ if 1, then parameter tracks are plotted

[pars,traj]=normmix_sgd(x,K,Niter,step0,plotflag); %uses gradient-descent optimization to estimate a Gaussian mixture model.
%pars : A structure containing the estimated parameters.
%traj : the parameter trajectories.

[cl,p]=normmix_classify(Istacked,pars);
I_classGMM = Istacked; 
I_classGMM(cl~=K,:)=256; 
figure(4)
subplot(1, 2, 1), imagesc(permeability)
title('Original image')
axis image;
subplot(1, 2, 2), imagesc(reshape(I_classGMM,[60 220]));
title('Segmentation using GMM with k=2')
axis image;
%% Markov random field mixture model
K=2;
figure(5)
N=[ 0 1 0 ; 1 0 1; 0 1 0]
opts = struct('plot',2,'N',N,'common_beta',1);
[theta,alpha,beta,cl,p] = mrf_sgd(permeability,K,opts);
I_classMRF = Istacked; 
I_classMRF(cl~=K,:)=256; 

figure(5)
subplot(1, 2, 1), imagesc(permeability)
title('Original image')
axis image;
subplot(1, 2, 2), imagesc(reshape(I_classMRF,[60 220]));
title('Segmentation using MRF with k=2')
axis image;
