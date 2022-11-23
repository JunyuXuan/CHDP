
clear;

%% read data


load data/multilabel/medical/medical_data_train.mat;


[A, D]  = size(DL_train');
V       = size(DF_train, 2);
AD      = DL_train';


Nd = zeros(1, D);

DN = sum(sum(DF_train));

dv_d_list = zeros(1, DN);
dv_v_list = zeros(1, DN);

id = 1;

for d = 1 : D
   
    Nd(d) = sum(DF_train(d, :));
    
    n_idx = find(DF_train(d, :) > 0);
    
    for n = 1 : length(n_idx)
    
        v   = n_idx(n);
        num = DF_train(d, v);
        
        dv_d_list(id:(id+num-1)) = d * ones(1, num);
        dv_v_list(id:(id+num-1)) = v * ones(1, num);
        
        id = id + num;
    end
end

clearvars n d v num n_idx N id;

%% run code
gam     = 0.5;
alpha_0 = 0.1;
alpha_a = 0.1;
alpha_d = 0.1;

Max_iteration = 2000;

% CHDP-S
[ K_list_irps, L_list_irps, CHDPS ] = chdps_irp_inference3( A, D, V, AD, Nd, dv_d_list, dv_v_list, alpha_0, alpha_a, alpha_d, gam, Max_iteration );

save convergenceML_irps_2000.mat K_list_irps L_list_irps;

% CHDP-M
[ K_list_irpm, L_list_irpm, CHDPM ] = chdpm_irp_inference3( A, D, V, AD, Nd, dv_d_list, dv_v_list, alpha_0, alpha_a, alpha_d, gam, Max_iteration );


%% plot L and K series and ACFs


save convergenceML_irpm_2000.mat K_list_irpm L_list_irpm;



