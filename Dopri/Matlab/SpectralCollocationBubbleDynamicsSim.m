function [t, x, y] = SpectralCollocationBubbleDynamicsSim(p, t_span, init_x, num)
% Function simulating bubble dynamics with spectral collocation and ode45
% Necessary parameters in p struct. Following are values for 25°C
% p.rho_L = 9.970639504998557e+02;
% p.p_inf = 1.0e+5;
% p.sigma = 0.071977583160056;
% p.R_E = 10e-6; %1...10u
% p.gamma = 1.33;
% p.c_L = 1.497251785455527e+03; % water 25 Celsius
% p.mu_L = 8.902125058209557e-04; %water
% p.lambda = 0.6084; %water 25 Celsius
% p.T_inf = 298.15; % 25 Celsius
% p.p_A = 1e5; % 0...2 bar
% p.f = 20e3; %20k...2M Hz

omega = 2*pi*p.f;

C.gamma = p.gamma;
C.C1 = omega*p.R_E/(2*pi*p.c_L);
pi2wRE = 2*pi/(omega*p.R_E);
C.C2 = 4*p.mu_L/(p.c_L*p.rho_L*p.R_E);
C.C3 = 4*p.mu_L/(p.rho_L*p.R_E)*pi2wRE;
C.C4 = 2*p.sigma*pi2wRE*pi2wRE/(p.rho_L*p.R_E);
C.C5 = p.p_inf/p.rho_L*pi2wRE*pi2wRE;
C.C6 = p.p_A/p.rho_L*pi2wRE*pi2wRE;
C.C7 = pi2wRE*p.p_inf/(p.c_L*p.rho_L);
C.C8 = pi2wRE*p.p_A/(p.c_L*p.rho_L);
C.C9 = 2*pi*pi2wRE*p.p_A/(p.c_L*p.rho_L);
C.C10 = p.lambda*(p.gamma-1)/p.gamma*pi2wRE/p.R_E*p.T_inf/p.p_inf;
C.C11 = p.lambda*(p.gamma-1)*pi2wRE/p.R_E*p.T_inf/p.p_inf;
C.C12 = (p.gamma-1)/p.gamma;
C.C13 = 1.0/(3.0*p.gamma);

y = 0; %to prevent errors

N = num; %number of collocation points, must be even
i = 0:1:(N-1);
y_full = cos(pi*i/(N-1))';

D = zeros(N,N);
p_i = 1;
p_j = 1;
for i = 1:1:N
    for j = 1:1:N
        if i==j
            if i==N
                D(N,N) = -(1+2*(N-1)*(N-1))/6;
            elseif i==1
                D(1,1) = (1+2*(N-1)*(N-1))/6;
            else
                D(i,i) = -y_full(i)/(2*(1-y_full(i)*y_full(i)));
            end
        else
           if i == 1 || i == N
               p_i = 2;
           else
               p_i = 1;
           end
           if j == 1 || j == N
               p_j = 2;
           else
               p_j = 1;
           end
           D(i,j) = (-1)^(i+j)*p_i/(p_j * (y_full(i)-y_full(j)) );
        end
    end
end

% [D2,x] = cheb_derivative(N-1);
% x
%  D
% D2

De = D(1:N/2,1:N/2);
Do = De;
for i = 1:N/2
    for j = 1:N/2
        De(i,j) = De(i,j) + D(i,N+1-j);
    end
end
for i = 1:N/2
    for j = 1:N/2
        Do(i,j) = Do(i,j) - D(i,N+1-j);
    end
end

y = y_full(1:N/2);
C.y = y;
C.y_sq = y.*y;
C.N = N;
C.De = De;
C.Do = Do;
options = odeset("RelTol",1e-10,"AbsTol",1e-10);
[t, x] = ode45(@(t,x) bubbleRightSide(t,x,C), t_span, init_x', options); 
end


%__________________________________________________________________________
function dxdt = bubbleRightSide(t, x, C)

    dxdt = x;
    rec_x1 = 1/x(1);
    rec_x3 = 1/x(3);
    sin2pit = sin(2*pi*t);
    
    dex = C.De*x(4:3+C.N/2);
    
    dxdt(3) = 3*rec_x1*(C.C11*rec_x1* dex(1) - C.gamma*x(2)*x(3));
    dxdt(4:3+C.N/2) = ( C.y*x(2)*rec_x1-C.C10*rec_x1*rec_x1*rec_x3* dex + C.C13*rec_x3*C.y*dxdt(3) ).* dex ...
        + C.C12*rec_x3*dxdt(3)*x(4:3+C.N/2) + C.C10*rec_x3*x(4:3+C.N/2)*rec_x1*rec_x1./(C.y_sq) .* C.Do*(C.y_sq.* dex);
    
    dxdt(1) = x(2);
    Den = x(1) - C.C1*x(1)*x(2) + C.C2;
    Nom = 0.5*C.C1*x(2)*x(2)*x(2)-1.5*x(2)*x(2)-C.C3*x(2)*rec_x1 - C.C4*rec_x1...
        + C.C5*x(3) - C.C5 - C.C6*sin2pit + C.C7*x(2)*x(3) - C.C7*x(2) - C.C8*x(2)*sin2pit...
        - C.C9*x(1)*cos(2*pi*t) + C.C7*x(1)*dxdt(3);
    dxdt(2) = Nom/Den;
    dxdt(4) = 0; %Boundary condition
    
end
