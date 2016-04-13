% Note: need to install pmtk3 from https://github.com/probml/pmtk3
% Run this script with Wholesale customers data.csv in the same directory
% Data can be obtained from http://archive.ics.uci.edu/ml/datasets/Wholesale+customers#


traindata = [];
testdata = [];

% Import data from CSV file, ignoring headers
csvdata = csvread('Wholesale customers data.csv', 1, 0);

% Select desired variables, Divide into training and test sets
rp = randperm(length(csvdata));

for c=1:length(csvdata)
    if mod(rp(c),2) == 0;
        traindata(end+1,:) = csvdata(rp(c),[1,3,7]);
    else
        testdata(end+1,:) = csvdata(rp(c),[1,3,7]);
    end
end

% Standardize columns
sctr(:,1) = standardizeCols(traindata(:,2));
sctr(:,2) = standardizeCols(traindata(:,3));
scte(:,1) = standardizeCols(testdata(:,2));
scte(:,2) = standardizeCols(testdata(:,3));

% Training and test labels
trla1 = find(traindata(:,1)==1);
trla2 = find(traindata(:,1)==2);
tela = testdata(:,1);

% Run GMM
sctr1 = sctr(trla1,:);
sctr2 = sctr(trla2,:);
[model1, loglikHist1] = mixGaussFit(sctr1, 1,'maxIter',10);
[model2, loglikHist2] = mixGaussFit(sctr2, 1,'maxIter',10);

logp1 = mixGaussLogprob(model1, scte);
logp2 = mixGaussLogprob(model2, scte);

% Assign predicted classes based on log likelihoods
for c=1:length(tela) 
    if (logp1(c)>logp2(c)) 
        class(c) = 1; 
    else
        class(c) = 2;
    end; 
end;

% Count matches between predicted and actual 'Channels', divide by number of elements to compute accuracy
count = 0;
for c=1:length(class) 
    if(class(c) == tela(c))
        count=count+1; 
    end; 
end;
disp(['Test Accuracy: ' num2str(100*count/length(tela)) '%']);

% Repeat on training data
logp1 = mixGaussLogprob(model1, sctr);
logp2 = mixGaussLogprob(model2, sctr);

trla = traindata(:,1);

for c=1:length(trla) 
    if (logp1(c)>logp2(c)) 
        class2(c) = 1; 
    else
        class2(c) = 2;
    end; 
end;

count = 0;
for c=1:length(class2) 
    if(class2(c) == trla(c))
        count=count+1; 
    end; 
end;
disp(['Training Accuracy: ' num2str(100*count/length(trla)) '%']);

% Plot data with channels
symbols = {'g.', 'bx'};
figure;
hold on;
plot(sctr(trla1,1), sctr(trla1,2), symbols{1}, 'markersize', 10);
plot(sctr(trla2,1), sctr(trla2,2), symbols{2}, 'markersize', 10);

title('Wholesale Customers Training Data')
xlabel('Fresh Food Sales')
ylabel('Detergent and Paper Sales')
legend('Horeca','Retail')

% Plot GMM output
symbols = {'g.', 'bx'};
symbols2 = {'r.', 'rx'};
hold off;
figure;
hold on;
for k=1:2
    match = find(tela'==k);
    nomatch = find(class~=tela' & tela'==k);
    plot(scte(match,1), scte(match,2), symbols{k}, 'markersize', 10);
    plot(scte(nomatch,1), scte(nomatch,2), symbols2{k}, 'markersize', 10);
end
title('Gaussian Mixture Model Output')
xlabel('Fresh Food Sales')
ylabel('Detergent and Paper Sales')
legend('Horeca, True','Horeca, False','Retail, True','Retail, False')

% Plot training output with error
symbols = {'g.', 'bx'};
symbols2 = {'r.', 'rx'};
hold off;
figure;
hold on;
for k=1:2
    match = find(trla'==k);
    nomatch = find(class2~=trla' & trla'==k);
    plot(sctr(match,1), sctr(match,2), symbols{k}, 'markersize', 10);
    plot(sctr(nomatch,1), sctr(nomatch,2), symbols2{k}, 'markersize', 10);
end
title('Training GMM Output')
xlabel('Fresh Food Sales')
ylabel('Detergent and Paper Sales')
legend('Horeca, True','Horeca, False','Retail, True','Retail, False')