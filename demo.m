% Read the input image. 
im = imread('winslet.jpg'); 
figure, imshow(im); 
title('Kate Winslet','fontsize',15); 
set(gcf, 'color','w'); 


% Draw the rectangular regions for negative and positive samples. The
% rectangles for these regions were manually selected using [ region, rect ] = imcrop. 
rect_positive  = [ 257.5100,  131.5100,   33.9800,  224.9800 ]; 
rect_negative = [ 49.5100,   92.5100,   51.9800,  164.9800 ];

hold on;
rectangle('Position',rect_positive,'edgecolor','b'); 
rectangle('Position',rect_negative,'edgecolor','r'); 



%%
% Get the image regions of these rectangular regions
positiveRegion  = imcrop( im, rect_positive ); 
negativeRegion = imcrop( im, rect_negative ); 

% Create the data vectors for positive and negative samples. 
% In this example, we take only the red channel of the image as measurement 
% for skin detection. 
N = 300;           % Use 300 points inside each region
r1 = positiveRegion( :, :, 1 ); 
r1 = r1(1:N)';                             % Re-shape the data to form a vector of measurements. 

r2 = negativeRegion( :, :, 1 ); 
r2 = r2(1:N)';                             % Re-shape the data to form a vector of measurements. 

r = [r1; r2];


% Attach an additional column to the data for the class assignments.
% The classes are 1 and 2, instead of 0 and 1, just a convenience
% for better workflow in Matlab because indexing begins at 1.

class_column = [ones(N,1); ones(N,1)+1 ];
%X = [ones(1,I_0+I_1); X];
x_data = [r class_column];
%x_train = 
x_train = [ones(1,size(x_data,1)); x_data'];


%class_column = [ones(N,1); ones(N,1)+1];
%x_train = [r class_column];
%x_train = double(x_train);

% Plot measurements. Plot negative values in blue.
figure;set(gcf, 'color','w'); 
y = zeros(N, 1);
plot (r1,y,'bo');
hold on
plot (r2,y,'ro');
axis([0,300,-10,100]);
title('Measurements (x)', 'FontSize', 14);
xlabel('x', 'FontSize', 16);


% Estimate the prior, p(w). This is a Bernoulli distribution, lambda.
W=2;
NTotal = size(x_train,1);               % Total number of measurements
for w = 1 : W 
    N(w) = sum( x_train(:,2)==w );   % Total number of measurements per class. 
    lambda(w) = N(w) / NTotal;                % Proportions for each class
end
lambda = lambda';                          % Transpose to make it a column vector 

% Plot lambda which represents P(w).
figure;
bar(lambda, 'y');
axis([0.5,2.5,0,1]);
xlabel('w', 'FontSize', 16);
ylabel('P(w)', 'FontSize', 16);
set(gca,'YTick',[0, 0.5, 1]);
title('(b)', 'FontSize', 14);


%x_test = -100 : 700;           
%X_test = 0:10:300;
%X_test = [ones(1,size(X_test,2)); X_test];
granularity = 100; 
a = 0;
b = 300;
domain = linspace (a, b, granularity);
[X,Y] = meshgrid (domain, domain);
x = X(:);
y = Y(:);
var_prior = 6;
x_test = [ones(1,granularity*granularity); x'; y'];
w = [zeros(granularity,1); ones(granularity,1)];

% Fit a logistic regression model.

initial_phi = [1; 1; 1];

[predictions, phi] = fit_logr (x_train, w, var_prior, x_test, initial_phi);

% Plot the predictions.
%plot(0:20:300, predictions, 'LineWidth',2,'LineSmoothing','on','Color','r');

% Plot the decision boundary.
%decision_boundary = -phi(1) / phi(2);
%plot(repmat(decision_boundary -10,100), linspace(-2,2,10),...
%'LineWidth',2,'LineSmoothing','on','Color','c');

%hold off;