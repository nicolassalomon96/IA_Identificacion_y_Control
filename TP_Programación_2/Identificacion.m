% Identificación de un sistema desconocido
clear all;
close all;
clc;

% Genero la señal de entrada al sistema
ts = 0.005; %5[ms]
N = 1000;
t = (0:N-1)*ts;
xt = randn(1,N);

plot(t,xt)
title('Señal aleatoria de entrada al sistema')
xlabel('Tiempo [s]')
grid on

%% ADALINE CON LMS
tic;
p = 3 ; %Retardos de la red == Dimensionalidad del sistema
b1 = [0.8 -0.2 0.5]; %Parametros del sistema durante t1 (Desconocidos)

%Para generar los retardos hacemos uso de la función de convolución en
%conjunto con el número de retardos
sd = length(t);
X = convmtx(xt, p);
X = X(:, 1:N);

d = b1*X(:,1:N);

y = zeros(size(sd)); %Salida predicha
error = zeros(size(sd)); %Error de predicción
lr = 0.01; %learning rate
w = rand(1, p); %Inicialización aleatorio del vector de pesos

for n = 1:N
    y(n) = w*X(:,n) ; %Señal predicha
    error(n) = d(n) - y(n) ; %Calculo del error
    w = w + 2*lr*error(n)*X(:,n)';
end
toc;

%% Graficas
figure(1)
subplot(2,1,1)
plot(t, xt)
grid on
title('Señal de entrada x(t)')
xlabel('Tiempo[s]')

subplot(2,1,2)
plot(t, d, '--b', t, y, '-r')
grid on
title('Salida real vs Salida predicha')
legend('Salida real','Salida predicha')
xlabel('Tiempo[s]')

figure(2)
plot(t, error)
grid on
title('Error de predicción')
xlabel('Tiempo[s]')

disp('Parametros reales durante t:');
disp(b1);
disp('Parametros predichos durante t:');
disp(w);

%% ADALINE CON Marquart
tic;
p = 3 ; %Retardos de la red == Dimensionalidad del sistema
b1 = [0.8 -0.2 0.5]; %Parametros del sistema durante t1 (Desconocidos)

%Para generar los retardos hacemos uso de la función de convolución en
%conjunto con el número de retardos
sd = length(t);
X = convmtx(xt, p);
X = X(:, 1:N);

d1 = b1*X(:,1:N); %Salida del sistema
d1 = con2seq(d1);
signal = con2seq(xt);

inputDelays   = 0:p-1; % Dado que necesitamos retrasar la señal en 3, necesitamos 2 bloques de retardos
learning_rate = 0.01; % learning rate

%ADALINE
net = linearlayer(inputDelays,learning_rate);
net.trainFcn = 'trainlm'; %Algoritmo de Levenberg-Marquardt
[net,Y,E] = adapt(net,signal,d1);

%ver arquitectura de la red
%view(net)

w1 = net.IW{1};
%transformar salida de datos secuenciales a vectores
Y = seq2con(Y); Y = Y{1};
E = seq2con(E); E = E{1};
d1 = seq2con(d1); d1 = d1{1};
toc;
%% Graficas
figure(3)
subplot(2,1,1)
plot(t, xt)
grid on
title('Señal de entrada x(t)')
xlabel('Tiempo[s]')

subplot(2,1,2)
plot(t, d1, '--b', t, Y, '-r')
grid on
title('Salida real vs Salida predicha')
legend('Salida real','Salida predicha')
xlabel('Tiempo[s]')

figure(4)
plot(t, E)
grid on
title('Error de predicción')
xlabel('Tiempo[s]')

disp('Parametros reales durante t:');
disp(b1);
disp('Parametros predichos durante t:');
disp(w1);