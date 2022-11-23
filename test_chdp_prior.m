clear

%% randomly generate data

% A = 200;
% D = 300;
% V = 400;
% 
% AD              = zeros(A,D);
% 
% for d = 1 : D
%    a        = find(mnrnd(1, 1/A * ones(1, A)) == 1);
%    AD(a, d) = 1; 
% end
% 
% DV              = binornd(1, 0.5*ones(D,V));
% 
% idx_v0          = find(sum(DV) == 0);
% DV(1, idx_v0)   = 1;
% 
% idx_d0          = find(sum(DV,2) == 0);
% DV(idx_d0, 1)   = 1;

%% run code

load chs_data4.mat;

alpha_0 = 1;
alpha_a = 1;
alpha_d = 1;

Max_iteration = 2000;

% CHDP-S
[ K_list_irps_prio ] = chdps_irp_prior( A, D, V, AD, DV, alpha_0, alpha_a, alpha_d, Max_iteration );

% CHDP-M
[ K_list_irpm_prio ] = chdpm_irp_prior( A, D, V, AD, DV, alpha_0, alpha_a, alpha_d, Max_iteration );

% In theory
[ expectation ] = chdp_expectation( A, D, AD, DV, alpha_0, alpha_a, alpha_d )
    
    figure
    subplot(2,2,1);
     
    tab1 = tabulate(K_list_irps_prio(500:end));
    bar(tab1(:, 1), tab1(:, 2));
    title('IRP-S-PRIOR');
         
    subplot(2,2,2);

    tab = tabulate(K_list_irpm_prio(500:end));  
    bar(tab(:, 1), tab(:, 2))
    title('IRP-M-PRIOR');



