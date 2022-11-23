clear

%% read data


% load data/authorpredict/NIPS/nips_predict_small.mat;
% 
% AD = full(AD_training);
% 
% V = N;
% 
% Nd = zeros(1, D_training);
% 
% DN = sum(sum(WD_training));
% 
% dv_d_list = zeros(1, DN);
% dv_v_list = zeros(1, DN);
% 
% id = 1;
% 
% for d = 1 : D_training
%    
%     Nd(d) = sum(WD_training(:, d));
%     
%     n_idx = find(WD_training(:, d) > 0);
%     
%     d
%     
%     for n = 1 : length(n_idx)
%     
%         v   = n_idx(n);
%         num = WD_training(v, d);
%         
%         dv_d_list(id:(id+num-1)) = d * ones(1, num);
%         dv_v_list(id:(id+num-1)) = v * ones(1, num);
%         
%         id = id + num;
%     end
% end
% 
% clearvars n d v num n_idx N;

load chs_realworlddata_test.mat;

%% run code

eta     = 0.5;
alpha_0 = 1;
alpha_a = 1;
alpha_d = 1;

Max_iteration = 500;

% VI-S
[ L_list_vis, change_list_vis, CHDPSV ] = chdps_vi_inference1( A, D, V, AD, Nd, dv_d_list, dv_v_list, alpha_0, alpha_a, alpha_d, eta, Max_iteration );


% VI-M
%[ L_list_vim, change_list_vim, CHDPMV ] = chdpm_vi_inference1( A, D, V, AD, Nd, dv_d_list, dv_v_list, alpha_0, alpha_a, alpha_d, eta, Max_iteration );


    figure
    subplot(2,2,1); 
    plot(change_list_vis);    
    title('VIS-change');
%     
    subplot(2,2,2);
    plot(L_list_vis);    
    title('VIS-L');

%     subplot(2,2,3); 
%     plot(change_vim_list);    
%     title('VIM-change');
%     
%     subplot(2,2,4);
%     plot(L_list_vim_infn);    
%     title('VIM-L');
    
% save 'irpoutput.mat' CHDPS CHDPM;
