%% TASK 3 - varying the percentage of missing pixels
% TITAN
% Y=imread('titan.jpg');
% [row col]=size(Y);

% ROSETTA
Y=imread('rosetta.jpg');
Y=rgb2gray(Y);
[row col color]=size(Y);
Y=Y(:,:,1);

% Convert the pixel values to double values between 0 and 1.
Y=double(Y)/255; 

% Y is the matrix of our pixel values so we are now choosing the indices that are
% supposed to be missing and the ones that are observed

perc=0.7; % the percentage of missing values
Y=reshape(Y, [1 row*col]);
rng(1); % set seed for randperm
shuffledIndices=randperm(row*col); % shuffled indices of the vector Y
missingIndices=shuffledIndices(1:perc*length(Y));
observedIndices=shuffledIndices(perc*length(Y)+1:end);
%%
% X is the matrix of all locations x_i, x_j (size length(Y)*2)
xi=1:row;
xj=1:col;
[Xi Xj]=meshgrid(xi, xj);
X=[Xi(:), Xj(:)]; % location matrix

%%
% here we are randomly choosing the observed points that we are going to
% use - both the locations and the values of the locations through a
% "working" index

if perc==0.5|perc==0.6
    n=10000;
else
    n=length(Y)*(1-perc);
end

workingIndices=observedIndices(1:n);
workingLocations=X(workingIndices,:);
workingValues=Y(workingIndices);
% Task 1: variograms and parameter estimation
% Empirical variogram
out=emp_variogram(workingLocations, workingValues,50);
fixed=struct('nu',1); % we want to use nu=1
params=cov_ls_est(workingValues,'matern',out, fixed)

% Matern variogram
estMaternVario= matern_variogram(out.h, params.sigma, params.kappa, params.nu, params.sigma_e);
% Plotting distances vs the variogram values 
figure(1)
hold on
plot(out.h, out.variogram, 'o') 
plot(out.h, estMaternVario)
xlabel('h')
ylabel('variogram')

i=i+1;

% TASK 2 - reconstructing the images using a stencil for precision images with varying kappa-values
observedValues=Y(observedIndices)';

% Kappa for task 3
kappaVector=params.kappa;

tau = 2*pi/(params.sigma^2);
k=kappaVector
q = (k^4)*[0 0 0 0 0; 0 0 0 0 0; 0 0 1 0 0; 0 0 0 0 0; 0 0 0 0 0] + (2*k^2)*[0 0 0 0 0; 0 0 -1 0 0; 0 -1 4 -1 0; 0 0 -1 0 0; 0 0 0 0 0] + [0 0 1 0 0; 0 2 -8 2 0; 1 -8 20 -8 1; 0 2 -8 2 0; 0 0 1 0 0];
Q = tau * stencil2prec([row col], q);

% B and beta values
B=[ones(length(observedIndices),1)];
betahat = (B'*B)\(B'*observedValues);
muhat=B*betahat;

% Precision matrices
Q21=Q(missingIndices, observedIndices);
Q22=Q(missingIndices, missingIndices);

% The Kriging predictor
A=(Q22\(Q21*(observedValues-muhat))); 
kriging= muhat(1).*ones(length(A),1) - A;

image0=Y;
image0(missingIndices)=kriging;

newImage=reshape(image0, [row col]);

hold on
figure(2)
subplot(1, 2, 1),imshow(reshape(Y, [row col])) 
title(['Original image'])
subplot(1, 2, 2),imshow(newImage)
title(['Image reconstruction using kappa=',num2str(k), ' and percentage of missing data is ',num2str(perc*100), '%']);
axis equal

