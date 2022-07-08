clear all
close all
clc

% Generación de la señal + ruido
t = [0:0.01:10]; %Tiempo
f = 1.0; %Frecuencia
x = sin(2*pi*f*t); %Señal sin ruido
n = 3.*(rand(1,length(t)) - 1); %ruido
y = n + x; %señal + ruido

figure(1)
subplot(3,1,1)
plot(t,x)
title('Señal sin ruido')
xlabel('Tiempo[s]')

subplot(3,1,2)
plot(t,n)
title('Ruido')
xlabel('Tiempo[s]')

subplot(3,1,3)
plot(t,y)
title('Señal con ruido')
xlabel('Tiempo[s]')


%% Algoritmo de cancelación de ruido
N = length(n);
r = 1; %Nº de retardos
X = convmtx(n,r); %Retardamos la señal con ruido r retardos
X = X(:,1:N);

w = rand(1,r); %Iniciamos pesos aleatorios
lr = 0.002; %learning rate
e = zeros(1,N);
y1 = zeros(1,N);

for k=1:N
    y1(k) = w*X(:,k);
    e(k) = y(k) - y1(k);
    w = w + lr*e(k)*X(:,k)';
end

%% Gráficas
figure(2)

subplot(4,1,1)
plot(t,n,'k')
title('Ruido')
grid on

subplot(4,1,2)
plot(t,x,'m')
title('Señal sin ruido')
grid on

subplot(4,1,3)
plot(t,y,'b')
title('Señal + Ruido')
grid on

subplot(4,1,4)
plot(t,e,'r')
title('Señal con ruido filtrado')
grid on

figure(3)

subplot(2,1,1)
plot(t,x,'--k','LineWidth',1)
hold all
plot(t,y,'r','LineWidth',.5)
legend('Señal sin ruido', 'Señal con ruido')
title('Señal sin ruido vs Señal con ruido')
grid on

subplot(2,1,2)
plot(t,x,'--k','LineWidth',1)
hold all
plot(t,e,'r','LineWidth',.5)
legend('Señal sin ruido', 'Señal con ruido filtrado')
title('Señal sin ruido vs Señal filtrada')
grid on