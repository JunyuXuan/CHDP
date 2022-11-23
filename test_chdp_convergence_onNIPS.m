
clear;

%% read data


load data/authorpredict/NIPS/nips_predict_small.mat;

AD = full(AD_training);

idx1 = find(WD_training > 0);

WD_training(idx1) = 1;

V = N;

Nd = zeros(1, D_training);

DN = sum(sum(WD_training));

dv_d_list = zeros(1, DN);
dv_v_list = zeros(1, DN);

id = 1;

for d = 1 : D_training
   
    Nd(d) = sum(WD_training(:, d));
    
    n_idx = find(WD_training(:, d) > 0);
    
    for n = 1 : length(n_idx)
    
        v   = n_idx(n);
        num = WD_training(v, d);
        
        dv_d_list(id:(id+num-1)) = d * ones(1, num);
        dv_v_list(id:(id+num-1)) = v * ones(1, num);
        
        id = id + num;
    end
end

clearvars n d v num n_idx N id idx1;


%% run code
gam     = 0.5;
alpha_0 = 0.1;
alpha_a = 0.1;
alpha_d = 0.1;

Max_iteration = 2000;

% CHDP-S
%[ K_list_irps, L_list_irps, CHDPS ] = chdps_irp_inference3( A, D_training, V, AD, Nd, dv_d_list, dv_v_list, alpha_0, alpha_a, alpha_d, gam, Max_iteration );

% CHDP-M
[ K_list_irpm, L_list_irpm, CHDPM ] = chdpm_irp_inference3( A, D_training, V, AD, Nd, dv_d_list, dv_v_list, alpha_0, alpha_a, alpha_d, gam, Max_iteration );


%% plot L and K series and ACFs


save convergenceNIPS_irpm.mat K_list_irpm L_list_irpm;



