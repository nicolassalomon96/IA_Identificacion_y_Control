clear all
close all
clc

% Generaci�n de la se�al + ruido
t = [0:0.01:10]; %Tiempo
f = 1.0; %Frecuencia
x = sin(2*pi*f*t); %Se�al sin ruido
n = 3.*(rand(1,length(t)) - 1); %ruido
y = n + x; %se�al + ruido

figure(1)
subplot(3,1,1)
plot(t,x)
title('Se�al sin ruido')
xlabel('Tiempo[s]')

subplot(3,1,2)
plot(t,n)
title('Ruido')
xlabel('Tiempo[s]')

subplot(3,1,3)
plot(t,y)
title('Se�al con ruido')
xlabel('Tiempo[s]')


%% Algoritmo de cancelaci�n de ruido
N = length(n);
r = 1; %N� de retardos
X = convmtx(n,r); %Retardamos la se�al con ruido r retardos
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

%% Gr�ficas
figure(2)

subplot(4,1,1)
plot(t,n,'k')
title('Ruido')
grid on

subplot(4,1,2)
plot(t,x,'m')
title('Se�al sin ruido')
grid on

subplot(4,1,3)
plot(t,y,'b')
title('Se�al + Ruido')
grid on

subplot(4,1,4)
plot(t,e,'r')
title('Se�al con ruido filtrado')
grid on

figure(3)

subplot(2,1,1)
plot(t,x,'--k','LineWidth',1)
hold all
plot(t,y,'r','LineWidth',.5)
legend('Se�al sin ruido', 'Se�al con ruido')
title('Se�al sin ruido vs Se�al con ruido')
grid on

subplot(2,1,2)
plot(t,x,'--k','LineWidth',1)
hold all
plot(t,e,'r','LineWidth',.5)
legend('Se�al sin ruido', 'Se�al con ruido filtrado')
title('Se�al sin ruido vs Se�al filtrada')
grid on