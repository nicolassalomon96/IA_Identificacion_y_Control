clear all
close all
clc

%% Graficamos la funcion a aproximar
x = 0:0.001:2*pi;
y = 2 * sin(20*pi*x) + 3*cos(10*pi*x) + sin(30*pi*x + pi/2);

plot(x,y)
xlim([0 2*pi])
title('Función a aproximar')

%% Creamos el dataset y definimos parametros
nsamples = 5000;
x_train = rand(nsamples,1) * 2*pi;
y_train = 2 * sin(20*pi*x_train) + 3*cos(10*pi*x_train) + sin(30*pi*x_train + pi/2);

%% Definimos la Red Neuronal
trainFcn = 'trainlm';  %lm Levenberg-Marquardt backpropagation

netconf = [50 25];
net = feedforwardnet(netconf, trainFcn);

%% Entrenamiento con 50 épocas
epochs = 50;
net.trainParam.epochs = epochs;

net_50 = train(net,x_train.',y_train.');

y2pred_50 = net_50(x);

figure
plot(x, y, '--', 'linewidth', 1);
hold all
plot(x, y2pred_50, 'linewidth', 1);
xlim([0 2*pi])
title('50 epochs fitting')

%% Entrenamiento con 100 épocas
epochs = 100;
net.trainParam.epochs = epochs;

net_100 = train(net,x_train.',y_train.');

y2pred_100 = net_100(x);

figure
plot(x, y, '--', 'linewidth', 1);
hold all
plot(x, y2pred_100, 'linewidth', 1);
xlim([0 2*pi])
title('100 epochs fitting')

%% Entrenamiento con 500 épocas
epochs = 500;
net.trainParam.epochs = epochs;

net_500 = train(net,x_train.',y_train.');

y2pred_500 = net_500(x);

figure
plot(x, y, '--', 'linewidth', 1);
hold all
plot(x, y2pred_500, 'linewidth', 1);
xlim([0 2*pi])
title('500 epochs fitting')