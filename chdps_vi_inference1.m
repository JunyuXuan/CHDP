% 
% CHDP with superposition 
%    
%    using stick breaking representation 
% 
%       by variational inference with data
%
%          note: data is represented by dv_d_list and dv_v_list  
%


function [ L_list, change_list, CHDPSV ] = chdps_vi_inference1( A, D, V, AD, Nd, d_list, v_list, alpha_0, alpha_a, alpha_d, eta, Max_iteration)

    L_list      = [];  
    change_list = [];
    
    % initialization
    [Ad, AO, DN, T, O, K, u_0, r_0, u_a, r_a, u_d, r_d, varsigma_ao, varsigma_dt, varsigma_dn, vartheta, gama] ...
                                = initialization(A, D, V, AD, Nd, d_list, v_list);
    
    % collect variables
    CHDPSV.K   = K;
    CHDPSV.u_0 = u_0;
    CHDPSV.r_0 = r_0;
    CHDPSV.u_a = u_a;
    CHDPSV.r_a = r_a;
    CHDPSV.u_d = u_d;
    CHDPSV.r_d = r_d;
    CHDPSV.varsigma_ao = varsigma_ao;
    CHDPSV.varsigma_dt = varsigma_dt;
    CHDPSV.varsigma_dn = varsigma_dn;
    CHDPSV.vartheta    = vartheta;       
    psi_vartheta       = psi(vartheta) - psi(repmat(sum(vartheta, 2), 1, V));
                                
    % iteration
    iter   = 1;
        
    while iter <= Max_iteration 
        
       fprintf(' --------iteration num = %d \n', iter);
       CHDPSV_old = CHDPSV;
       
       % update d level
       fprintf('             update d level  \n');
       tic; 
       
       % udpate varsigma_dn 
       fprintf('             update d level: varsigma_dn   \n');   
       parfor dv = 1 : DN
           
           varsigma_dn_tmp = varsigma_dn(dv, :);
           
           d   = d_list(dv);
           v   = v_list(dv);  
           tmp = zeros(1, T);
           
           for t = 1 : T
               tmp(t) = update_varsigma_dn(varsigma_dn_tmp(t), ...
                            d, v, t, T, AO, V, K, u_d, r_d, ...
                            varsigma_ao, varsigma_dt, psi_vartheta, gama);
           end
           
           varsigma_dn_tmp    = 1 ./ sum(exp(repmat(tmp, [T 1]) - repmat(tmp', [1 T])), 2);           
           varsigma_dn(dv, :) =  varsigma_dn_tmp;                              
       end
       
       fprintf('                                  use time : %d  \n', toc);
       fprintf('             update d level: u_d and r_d  \n');
       tic; 
       
       parfor d = 1 : D
           
           A_d = Ad(d);
           a_d = find(AD(:, d) > 0);
           
           % udpate u_d and r_d
           
           d_list_d       = find(d_list == d);
           varsigma_dn_d  = varsigma_dn(d_list_d, :);
           
           u_d(d, :)      = update_u_d(u_d(d, :), T, varsigma_dn_d, gama);
           r_d(d, :)      = update_r_d(r_d(d, :), T, varsigma_dn_d, gama, alpha_d);
                      
           % udpate varsigma_dt
           %fprintf('             update d level: varsigma_dt of %d  \n', d);
           varsigma_dt(d, :, :) = update_varsigma_dt(a_d, A_d, v_list(d_list_d), length(d_list_d), A, O, ...
               T, V, K, AO, u_a, r_a, psi_vartheta, varsigma_dn_d, reshape(varsigma_dt(d, :, :), T, AO), varsigma_ao, gama);
       end
       
       fprintf('                                  use time : %d  \n', toc);
               
       % update a level
       fprintf('             update a level  \n');
       tic;
       
       parfor a = 1 : A
           
           d_a       = find(AD(a, :) > 0);
           
           % udpate u_a and r_a 
           %fprintf('             update a level: u_%d and r_%d  \n', a, a);           
           u_a(a, :) = update_u_a(u_a(a, :), a, A, O, varsigma_dt, gama);
           r_a(a, :) = update_r_a(r_a(a, :), a, A, O, varsigma_dt, alpha_a, gama);
           
           % udpate varsigma_ao
           %fprintf('             update a level: varsigma_ao for %d \n', a);
           varsigma_ao(a, :, :) = update_varsigma_ao(reshape(varsigma_ao(a, :, :), O, K), a, d_a, T, A, O, K, V, ...
                                    d_list, v_list, u_0, r_0, varsigma_dn, varsigma_dt, psi_vartheta, gama);
       end
       
       fprintf('                                  use time : %d  \n', toc);

       % update k level
       fprintf('             update k level  \n');
       tic;
       
       % udpate u_0 and r_0 
       fprintf('             update k level: u_0 and r_0  \n');
       tic;
       u_0        = update_u_0(u_0, K, varsigma_ao, gama);
       r_0        = update_r_0(r_0, K, varsigma_ao, alpha_0, gama);
       fprintf('                                  use time : %d  \n', toc);
        
       % udpate vartheta
       fprintf('             update k level: vartheta  \n');
       tic;
       vartheta   = update_vartheta(vartheta, K, V, T, AO, d_list, v_list, ...
                                    varsigma_dt, varsigma_ao, varsigma_dn, eta, gama);
                                
       psi_vartheta  = psi(vartheta) - psi(repmat(sum(vartheta, 2), 1, V));
        
       fprintf('                                  use time : %d  \n', toc);
       
%        % udpate alpha_d
%        stepsize    = 0.01;
%        
%        alpha_d_new = alpha_d + stepsize*sum(sum(psi(1+alpha_d) - psi(alpha_d) + psi(r_d) - psi(u_d + r_d)));
% 
%        while alpha_d_new < 0
%            stepsize = stepsize * 0.1;
%            alpha_d_new = alpha_d + stepsize*sum(sum(psi(1+alpha_d) - psi(alpha_d) + psi(r_d) - psi(u_d + r_d)));
%        end
%        
%        alpha_d = alpha_d_new;
       
       % evaluate ELBO
       fprintf('             evaluate ELBO  \n');
       tic;
       L         = evaluate_likelihd(K, D, V, A, T, O, AO, DN, Ad, d_list, v_list, u_0, r_0, u_a, r_a, u_d, r_d, ...
                             varsigma_dn, varsigma_dt, varsigma_ao, vartheta, psi_vartheta, alpha_0, alpha_a, alpha_d, eta );
       fprintf('                                  L = %d,  use time : %d  \n', L, toc);
       
       %% output 
       L_list(iter)      = L;
       
       CHDPSV.L   = L;
       CHDPSV.K   = K;
       CHDPSV.u_0 = u_0;
       CHDPSV.r_0 = r_0;
       CHDPSV.u_a = u_a;
       CHDPSV.r_a = r_a;
       CHDPSV.u_d = u_d;
       CHDPSV.r_d = r_d;
       CHDPSV.varsigma_ao = varsigma_ao;
       CHDPSV.varsigma_dt = varsigma_dt;
       CHDPSV.varsigma_dn = varsigma_dn;
       CHDPSV.vartheta    = vartheta;
       
       change      = evaluate_variable_change(CHDPSV_old, CHDPSV)      
       change_list = [change_list change];
       
       if change < 0.001
          break; 
       end
       
       iter        = iter + 1;
        
    end
    
end


function [Ad, AO, DN, T, O, K, u_0, r_0, u_a, r_a, u_d, r_d, varsigma_ao, varsigma_dt, varsigma_dn, vartheta, gama] ...
    = initialization(A, D, V, AD, Nd, d_list, v_list)

    % Max number of iteration
%     Max_iteration = 5000;
        
    % number of authors of douments
    Ad   =  sum(AD ~= 0, 1);
    
    % number of words of douments
    %Nd   =  sum(DV ~= 0, 2);
    
    DN   = sum(Nd);
    
    % data list    
%     dn_list = find(DV > 0);
%     [d_list, v_list] = ind2sub([D V], dn_list);
        
    % proximal parameter
    gama = 0.1;
    
               
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% variables
    
    % truncations
    K        = 200;
    O        = 100;
    T        = 50;    
    AO       = A * O;
    
    % variational parameters for stick breaks
    u_0    = ones(1, K-1);
    r_0    = ones(1, K-1);
    u_a    = ones(A, O-1); 
    r_a    = ones(A, O-1); 
    u_d    = ones(D, T-1);
    r_d    = ones(D, T-1);
    
    % varitional parameters for indicator
    varsigma_ao = ones(A, O, K) / K;
    varsigma_dn = ones(DN, T) / T;
    varsigma_dt = zeros(D, T, AO);
    
    for d = 1 : D
       a_d = find(AD(:, d) > 0);
       A_d = Ad(d);
       
       ao_idx                    = sub2ind([A O], reshape(repmat(a_d, 1, O), 1, A_d*O), reshape(repmat(1:O, A_d, 1), 1, A_d*O));       
       varsigma_dt(d, :, ao_idx) = 1 / length(ao_idx);
    end
            
    % varitional parameters for topics
    vartheta  = ones(K, V);
    
end

function varsigma_dnt = update_varsigma_dn(varsigma_dnt_i, d, v, t, T, AO, V, K, u_d, r_d, varsigma_ao, varsigma_dt, psi_vartheta, gama)

    varsigma_dnt = 0;
    
    if t < T        
        varsigma_dnt = psi(u_d(d,t)) - psi(u_d(d,t) + r_d(d,t)) ;
    end    
    
    varsigma_dnt = varsigma_dnt + sum(psi(r_d(d,1:t-1)) - psi(u_d(d,1:t-1) + r_d(d,1:t-1)))...
                    - (1+gama) + gama*log(varsigma_dnt_i+eps);
               
    varsigma_dnt = varsigma_dnt + sum(sum(reshape(varsigma_ao, [AO K])  ...
                   .* repmat(transpose(reshape(varsigma_dt(d,t,:), [1 AO])), [1 K])...
                   .* repmat(transpose(psi_vartheta(:, v)), AO, 1)   ));
               
    varsigma_dnt = varsigma_dnt/(1+gama) ;         
end

function u_d = update_u_d(u_d_i, T, varsigma_dn_d, gama)

    u_d = (sum(varsigma_dn_d(:, 1:T-1), 1) + gama * (u_d_i-1))/(1+gama) + 1;

end

function r_d = update_r_d(r_d_i, T, varsigma_dn_d, gama, alpha_d)
    r_d = zeros(1, T-1);
    for t = 1 : T-1
        r_d(t) = (alpha_d - 1 + sum(sum(varsigma_dn_d(:,t+1:T))) + gama * (r_d_i(t)-1))/(1+gama) + 1;
    end
end

function u_0 = update_u_0(u_0_i, K, varsigma_ao, gama)

    u_0 = ( reshape(sum(sum(varsigma_ao(:, :, 1:K-1), 1), 2), 1, K-1) + gama * (u_0_i - 1))/(1+gama) + 1;

end

function r_0 = update_r_0(r_0_i, K, varsigma_ao, alpha_0, gama)

    r_0 = zeros(1, K-1);
    parfor k = 1 : K-1
        r_0(k) = ( alpha_0 - 1 + sum(sum(sum(varsigma_ao(:, :, k+1:K), 1), 2), 3) + gama*(r_0_i(k)-1) )/(1+gama) + 1;
    end
end

function vartheta  = update_vartheta(vartheta_i, K, V, T, AO, ...
    d_list, v_list, varsigma_dt, varsigma_ao, varsigma_dn, eta, gama)

    vartheta = zeros(K, V);
    
    parfor k = 1 : K
        varsigma_aok = transpose(reshape(varsigma_ao(:, :, k), AO, 1));
        
        for v = 1 : V
            
            v_list_v  = find(v_list == v);
            d_list_v  = d_list(v_list_v);
            %dn_list_v = dn_list(v_list_v);
            DNV       = length(v_list_v);
            
%             tmp = sum(sum(sum(reshape(repmat(varsigma_aok, DNV*T, 1), DNV, T, AO) ...
%                     .* varsigma_dt(d_list_v, :, :)...
%                     .* repmat(varsigma_dn(v_list_v, :), 1, 1, AO) ))) ;
            
            tmp = 0;
            for t = 1 : T               
                tmp = tmp+ sum(repmat(varsigma_aok', DNV, 1) .* reshape(varsigma_dt(d_list_v, t, :), DNV, AO) * varsigma_dn(v_list_v, t) );                
            end
            
            vartheta(k, v) = (eta + gama*vartheta_i(k, v) + tmp ) / (1+gama) ;
        end  
        
    end

end

%%%%%%%%%%%%%%%%%%%%%% the following is unique for CHDPS 

function varsigma_dt_d = update_varsigma_dt(a_d, A_d, dn_v_list, dn_length, A, O, ...
    T, V, K, AO, u_a, r_a, psi_vartheta, varsigma_dn_d, varsigma_dt_i, varsigma_ao, gama)

    varsigma_dt_d = zeros(T, AO);
    
    psi_vartheta_tmp = transpose(psi_vartheta(:, dn_v_list));
    
    
    for t = 1 : T
        
        p      = ones(1, A_d*O);
        ao_idx = zeros(1, A_d*O);
        varsigma_dn_d_tmp = repmat(varsigma_dn_d(:, t), 1, K);
        
        for aoi = 1 : A_d*O
            
            [ai, o]      = ind2sub([A_d O], aoi);
            a            = a_d(ai);            
            ao           = sub2ind([A O], a, o);
            ao_idx(aoi)  = ao;
            
            tmp = 0;
            
            if o < O
              tmp     = psi(u_a(a,o)) - psi(u_a(a,o) + r_a(a,o));
            end
            
%             tmp     = tmp +  sum(psi(r_a(a,1:o-1)) - psi(u_a(a,1:o-1) + r_a(a,1:o-1))) - log(A_d+eps)...
%                         - (1 + gama) + gama * log(varsigma_dt_i(t, ao)+eps)...
%                         + sum(sum(repmat(reshape(varsigma_ao(a, o, :), 1, K), dn_length, 1) .* repmat(varsigma_dn_d(:, t), 1, K) ...
%                             .* transpose(psi_vartheta(:, dn_v_list)) ));
            tmp     = tmp +  sum(psi(r_a(a,1:o-1)) - psi(u_a(a,1:o-1) + r_a(a,1:o-1))) - log(A_d+eps)...
                        - (1 + gama) + gama * log(varsigma_dt_i(t, ao)+eps)...
                        + sum(sum(repmat(reshape(varsigma_ao(a, o, :), 1, K), dn_length, 1) .*  varsigma_dn_d_tmp...
                            .* psi_vartheta_tmp ));
                        
            p(aoi) = tmp/(1+gama);
        end
        
        p = 1 ./ sum(exp(repmat(p, [A_d*O 1]) - repmat(p', [1 A_d*O])), 2);
        
        varsigma_dt_d(t, ao_idx) = p;
    end
end

function ua = update_u_a(ua_i, a, A, O, varsigma_dt, gama)
    
    ao_list  = sub2ind([A O], a*ones(1, O-1), 1:O-1);    
    ua       = (reshape(sum(sum(varsigma_dt(:, :, ao_list), 1), 2), 1, O-1) + gama * (ua_i-1))/(1+gama) + 1;
    
end

function ra = update_r_a(ra_i, a, A, O, varsigma_dt, alpha_a, gama)

    ra = zeros(1, O-1);

    for o = 1 : O-1        
        ah_list  = sub2ind([A O], a*ones(1, O-o), o+1:O);         
        tmp      = sum(sum(sum(varsigma_dt(:, :, ah_list) )));                
        ra(o)    = (alpha_a - 1 + tmp + gama * (ra_i(o)-1)) / (1+gama) + 1;        
    end

end

function varsigma_ao_a = update_varsigma_ao(varsigma_ao_a_i, a, d_a, T, A, O, K, V, ...
    d_list, v_list, u_0, r_0, varsigma_dn, varsigma_dt, psi_vartheta, gama)

    varsigma_ao_a = zeros(O, K);    
    %psi_vartheta  = psi(vartheta) - psi(repmat(sum(vartheta, 2), 1, V)); % K x V
    d_list_a      = ismember(d_list, d_a);
    %dn_list_a     = dn_list(d_list_a);
    v_list_a      = v_list(d_list_a);  
    d_list_a      = d_list(d_list_a);        
    DNA           = length(d_list_a);
        
    for o = 1 : O
        
        ao = sub2ind([A O], a, o);        
        p  = zeros(1, K);
        
        for k = 1 : K
                 
            tmp  = sum(sum(reshape(varsigma_dt(d_list_a, :, ao), DNA, T) .* varsigma_dn(d_list_a, :) .* repmat(transpose(psi_vartheta(k, v_list_a)), 1, T)));
            
            if k < K                
                tmp  = tmp + psi(u_0(k)) - psi(u_0(k) + r_0(k)) ;
            end
            
            tmp  = tmp + sum(psi(r_0(1:k-1)) - psi(u_0(1:k-1) + r_0(1:k-1)))...
                        - (1 + gama) + gama * log(varsigma_ao_a_i(o, k)+eps) ;
            
            p(k) = tmp/(1+gama);
        end
        
        varsigma_ao_a(o, :) = 1 ./ sum(exp(repmat(p, [K 1]) - repmat(p', [1 K])), 2);
    end
end

function L = evaluate_likelihd(K, D, V, A, T, O, AO, DN, Ad, d_list, v_list, u_0, r_0, u_a, r_a, u_d, r_d, ...
    varsigma_dn, varsigma_dt, varsigma_ao, vartheta, vartheta_tmp, alpha_0, alpha_a, alpha_d, eta )

    % theta
    L = K*gammaln(eta*V) - K*V*gammaln(eta) + (eta-1) * sum(sum(vartheta_tmp));
    
    L = L - (sum(gammaln(sum(vartheta, 2))) - sum(sum(gammaln(vartheta))) +  sum(sum((vartheta - 1) .* vartheta_tmp )) );

    % nu_0
    L = L + sum(gammaln(1+alpha_0) - gammaln(alpha_0) + (alpha_0-1) * (psi(r_0) - psi(u_0 + r_0)) );
    
    L = L - sum(gammaln(u_0+r_0) - gammaln(u_0) - gammaln(r_0) + (u_0-1) .* (psi(u_0) - psi(u_0+r_0)) + (r_0-1) .* (psi(r_0) - psi(u_0+r_0)) );
    
    % nu_a
    L = L + sum(sum(gammaln(1+alpha_a) - gammaln(alpha_a) + (alpha_a-1) * (psi(r_a) - psi(u_a+r_a)) ));
    
    L = L - sum( sum( gammaln(u_a+r_a) - gammaln(u_a) - gammaln(r_a) + (u_a-1) .* (psi(u_a) - psi(u_a+r_a)) + (r_a-1) .* (psi(r_a) - psi(u_a+r_a)) )); 
    
    % z_ao
    for k = 1 : K 
        if k < K
            tmp = psi(u_0(k)) - psi(u_0(k)+r_0(k));
        else
            tmp = 0;
        end
        tmp = tmp + sum(psi(r_0(1:k-1)) - psi(u_0(1:k-1) + r_0(1:k-1)));        
        L   = L + sum(sum(tmp * varsigma_ao(:, :, k)));        
    end
    
    L = L - sum(sum(sum( varsigma_ao .* log(varsigma_ao + eps) )));
    
    % nu_d
    L = L + sum(sum(gammaln(1+alpha_d) - gammaln(alpha_d) + (alpha_d-1) *(psi(r_d) - psi(u_d+r_d))));
    
    L = L - sum(sum(gammaln(u_d+r_d) - gammaln(u_d) - gammaln(r_d) + (u_d-1).*(psi(u_d) - psi(u_d + r_d)) + (r_d-1).*(psi(r_d) - psi(u_d+r_d)) ));
    
    % z_dt
    for ao = 1 : AO
       
        [a, o] = ind2sub([A O], ao);
        
        if o < O
            tmp = psi(u_a(a,o)) - psi(u_a(a,o)+r_a(a,o));
        else
            tmp = 0;
        end
        
        tmp = tmp + sum(psi(r_a(a, 1:o-1)) - psi(u_a(a, 1:o-1) + r_a(a, 1:o-1))) ;
        
        for d = 1 : D
            L = L + sum(varsigma_dt(d, :, ao) * (tmp - log(Ad(d))));
        end
    end
    
    L = L - sum(sum(sum(varsigma_dt .* log(varsigma_dt + eps) )));
    
    % z_dn
    for dn = 1 : DN
        d = d_list(dn);
        v = v_list(dn);
        
        for t = 1 : T            
            if t == T
                L = L + varsigma_dn(dn, t) * sum(psi(r_d(d, 1:t-1)) - psi(u_d(d, 1:t-1) + r_d(d, 1:t-1)));
            else
                L = L + varsigma_dn(dn, t) * (psi(u_d(d,t)) - psi(u_d(d,t) + r_d(d,t)) + sum(psi(r_d(d, 1:t-1)) - psi(u_d(d, 1:t-1) + r_d(d, 1:t-1))));
            end
            L = L + sum(sum(reshape(varsigma_ao, AO, K) .* repmat(reshape(varsigma_dt(d, t, :), AO, 1), 1, K) * varsigma_dn(dn, t) .* repmat(reshape(vartheta_tmp(:, v), 1, K), AO, 1) ));
        end
    end
    
    L = L - sum(sum(varsigma_dn .* log(varsigma_dn + eps) ));
end

function change = evaluate_variable_change(CHDPSV_old, CHDPSV)

    change = mean(abs(CHDPSV_old.u_0 - CHDPSV.u_0)) + mean(abs(CHDPSV_old.r_0 - CHDPSV.r_0));
    change = change + mean(mean(abs(CHDPSV_old.u_a - CHDPSV.u_a))) + mean(mean(abs(CHDPSV_old.r_a - CHDPSV.r_a))); 
    change = change + mean(mean(abs(CHDPSV_old.u_d - CHDPSV.u_d))) + mean(mean(abs(CHDPSV_old.r_d - CHDPSV.r_d)));
    
    change = change + mean(mean(abs(CHDPSV_old.varsigma_dn - CHDPSV.varsigma_dn))) ;
    change = change + mean(mean(abs(CHDPSV_old.vartheta - CHDPSV.vartheta)));
    change = change + mean(mean(mean(abs(CHDPSV_old.varsigma_dt - CHDPSV.varsigma_dt))));
    change = change + mean(mean(mean(abs(CHDPSV_old.varsigma_ao - CHDPSV.varsigma_ao))));
    
    change = change / 10;
end

