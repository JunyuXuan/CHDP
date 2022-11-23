clear

%% randomly generate data

% A = 20;
% D = 30;
% V = 50;
% 
% AD              = binornd(1, 0.3*ones(A,D));
% 
% idx_d0          = find(sum(AD) == 0);
% AD(1, idx_d0)   = 1;
% 
% idx_a0          = find(sum(AD,2) == 0);
% AD(idx_a0, 1)   = 1;
% 
% DV              = binornd(1, 0.5*ones(D,V));
% 
% idx_v0          = find(sum(DV) == 0);
% DV(1, idx_v0)   = 1;
% 
% idx_d0          = find(sum(DV,2) == 0);
% DV(idx_d0, 1)   = 1;



%% run code


load chs_data.mat;
%load chs_data1.mat;

K_alpha_a = zeros(4, 10, 10);

for alpha_a = 1 : 10
    
alpha_0 = 1;
%alpha_a = 1;
alpha_d = 1;
   
Max_iteration = 1000;

for num = 1 : 10
 
% CHDP-S
 [ K_list_irps_simu ] = chdps_irp_simulation( A, D, V, AD, DV, alpha_0, alpha_a, alpha_d, Max_iteration );

% CHDP-M
 [ K_list_irpm_simu ] = chdpm_irp_simulation( A, D, V, AD, DV, alpha_0, alpha_a, alpha_d, Max_iteration );

% CHDP-S
 [ K_list_sticks_simu ] = chdps_stick_simulation( A, D, V, AD, DV, alpha_0, alpha_a, alpha_d, Max_iteration );
% [ K_list_sticks_simu1 ] = chdps_stick_simulation1( A, D, V, AD, DV, alpha_0, alpha_a, alpha_d, Max_iteration );

% CHDP-M
[ K_list_stickm_simu ] = chdpm_stick_simulation( A, D, V, AD, DV, alpha_0, alpha_a, alpha_d, Max_iteration );
%[ K_list_stickm_simu1 ] = chdpm_stick_simulation1( A, D, V, AD, DV, alpha_0, alpha_a, alpha_d, Max_iteration );

% In theory
[ expectation ] = chdp_expectation( A, D, AD, DV, alpha_0, alpha_a, alpha_d )
    
tab1 = tabulate(K_list_irps_simu);
[~, idx] = max(tab1(:, 2));

K_alpha_a(1, alpha_a, num) = tab1(idx, 1);

tab1 = tabulate(K_list_irpm_simu);
[~, idx] = max(tab1(:, 2));

K_alpha_a(2, alpha_a, num) = tab1(idx, 1);

tab1 = tabulate(K_list_sticks_simu);
[~, idx] = max(tab1(:, 2));

K_alpha_a(3, alpha_a, num) = tab1(idx, 1);

tab1 = tabulate(K_list_stickm_simu);
[~, idx] = max(tab1(:, 2));

K_alpha_a(4, alpha_a, num) = tab1(idx, 1);

end

%     figure
%     subplot(2,2,1);
%      
%     tab1 = tabulate(K_list_irps_simu);      
%     bar(tab1(:, 1), tab1(:, 2));
%     title('IRP-S-SIMU');
% %         
%     subplot(2,2,2);
% 
%     tab2 = tabulate(K_list_irpm_simu);
%     bar(tab2(:, 1), tab2(:, 2))
%     title('IRP-M-SIMU');
% %    
%     subplot(2,2,3);   
% 
%     tab3 = tabulate(K_list_sticks_simu);
%     bar(tab3(:, 1), tab3(:, 2))
%     title('STICK-S-SIMU'); 
% % %     
%     subplot(2,2,4);    
% 
%     tab4 = tabulate(K_list_stickm_simu); 
%     bar(tab4(:, 1), tab4(:, 2))
%     title('STICK-M-SIMU'); 
 
end
    
save chdp_expect_a01_ad1.mat K_alpha_a;




