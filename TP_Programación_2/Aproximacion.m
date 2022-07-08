% Aproximación de función por ADALINE
clear;
clc;
% Señal de entrada
t = 0:0.01:30;
x = sin(t + sin(t.^2));
N = length(t); %longitud de muestras de tiempo

figure(1)
plot(t, x)
grid
title('Señal a aproximar')
xlabel('Tiempo [s]')
%% 3 Pasos de muestreo

p1 = 3 ; % Cantidad de retardos

X1 = convmtx(x, p1);
X1 = X1(:, 1:N); % Crea los retardos haciendo la convolución entre la señal de entrada y el número de retardos
d1 = x; %d1 = señal deseada a la salida. Es igual a la señal de entrada para este problema

y1 = zeros(size(d1));
error1 = zeros(size(d1));
learning_rate = 0.1; % learning rate
w1 = rand(1, p1) ; % Vector de pesos

for n = 1:N % Loop de entrenamiento
y1(n) = w1*X1(:,n) ; % Señal de salida predicha
error1(n) = d1(n) - y1(n) ; %Error
w1 = w1 + learning_rate*error1(n)*X1(:,n)';
end

figure(2)
subplot(2,1,1)
plot(t, d1, 'b', t, y1, '-r')
legend('Señal original','Predicción')
grid on
title('Señal original vs Señal predicha - 3 pasos')
xlabel('Tiempo[s]')

subplot(2,1,2)
plot(t, error1)
grid on
title('Error de predicción')
xlabel('Tiempo[s]')

%% 5 Pasos de muestreo
p2 = 5 ; % Cantidad de retardos
X2 = convmtx(x, p2) ; X2 = X2(:, 1:N); % Crea los retardos haciendo la convolución entre la señal de entrada y el número de retardos
d2 = x ; %d2 = señal deseada a la salida. Es igual a la señal de entrada para este problema

y2 = zeros(size(d2));
error2 = zeros(size(d2));
learning_rate = 0.1; % learning rate
w2 = rand(1, p2); % Vector de pesos

for n = 1:N % Loop de entrenamiento
y2(n) = w2*X2(:,n); % Señal de salida predicha
error2(n) = d2(n) - y2(n) ; %Error
w2 = w2 + learning_rate*error2(n)*X2(:,n)';
end

figure(3)
subplot(2,1,1)
plot(t, d2, 'b', t, y2, '-r')
legend('Señal original','Predicción')
grid on
title('Señal original vs Señal predicha - 5 pasos')
xlabel('Tiempo[s]')

subplot(2,1,2)
plot(t, error2)
grid on
title('Error de predicción')
xlabel('Tiempo[s]')