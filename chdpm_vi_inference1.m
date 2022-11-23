% 
% CHDP with Maximization 
%    
%    using stick breaking representation 
% 
%       by variational inference with data
%
%          note: data is represented by a binary matrix DV  
%
%               

function [ L_list, change_list, CHDPMV ] = chdpm_vi_inference1( A, D, V, AD, DV, alpha_0, alpha_a, alpha_d, eta, Max_iteration)

    L_list      = [];
    change_list = [];
    
    % initialization
    [Nd, Ad, AO, DN, S, dn_list, d_list, v_list, T, O, K, u_0, r_0, u_a, r_a, u_d, r_d, ...
        varsigma_ao, varsigma_dt, varsigma_dn, vartheta, gama, stepalpha] = initialization(A, D, V, AD, DV);
   
    % collect variables
    CHDPMV.u_0         = u_0;
    CHDPMV.r_0         = r_0;
    CHDPMV.u_a         = u_a;
    CHDPMV.r_a         = r_a;
    CHDPMV.u_d         = u_d;
    CHDPMV.r_d         = r_d;
    CHDPMV.varsigma_ao = varsigma_ao;
    CHDPMV.varsigma_dt = varsigma_dt;
    CHDPMV.varsigma_dn = varsigma_dn;
    CHDPMV.vartheta    = vartheta;    
    psi_vartheta       = psi(vartheta) - psi(repmat(sum(vartheta, 2), 1, V));
    
    % iteration
    iter   = 1;
        
    % sample nu_ao and z_ao    
    [nu_a_s, pi_a_s, z_a_s] = sample_a2(S, A, O, K, u_a, r_a, varsigma_ao);    
    Omega = cell(1, D);    
    for d = 1 : D
        a_d       = find(AD(:, d) > 0);
        A_d       = Ad(d);
        Omega_d   = evaluate_omega(pi_a_s(:, a_d, :), z_a_s(:, a_d, :), A_d, O, S, K);
        Omega{d}  = Omega_d;
    end
    
    while iter <= Max_iteration
       
       fprintf(' --------iteration num = %d \n', iter);        
       CHDPMV_old = CHDPMV;
       step       = iter^(-stepalpha);
                          
       % update d level
       fprintf('             update d level  \n');
       tic; 
       
       % udpate varsigma_dn 
       %fprintf('             update d level: varsigma_dn   \n'); 
       for dv = 1 : DN
                                
           d   = d_list(dv);
           v   = v_list(dv); 
           tmp = zeros(1, T);
           
           for t = 1 : T
               tmp(t) = update_varsigma_dn(varsigma_dn(dv,t), d, v, t, T, AO, V, K,...
                             u_d, r_d, varsigma_ao, varsigma_dt, psi_vartheta, gama);
           end
           
           varsigma_dn(dv, :) = 1 ./ sum(exp(repmat(tmp, [T 1]) - repmat(tmp', [1 T])), 2);                                
       end
       
       for d = 1 : D
           
           a_d = find(AD(:, d) > 0);
           A_d = Ad(d);
           
           % udpate u_d and r_d
           %fprintf('             update d level: u_%d and r_%d  \n', d, d);
           d_list_d       = find(d_list == d);
           varsigma_dn_d  = varsigma_dn(d_list_d, :);
           
           u_d(d, :)      = update_u_d(u_d(d, :), T, varsigma_dn_d, gama);
           r_d(d, :)      = update_r_d(r_d(d, :), T, varsigma_dn_d, gama, alpha_d);
                
           % udpate varsigma_dt
           %fprintf('             update d level: varsigma_dt of %d  \n', d);                      
           varsigma_dt(d, :, :)   = update_varsigma_dt(a_d, A_d, v_list(d_list_d), length(d_list_d),...
                                                       A, O, T, K, AO, psi_vartheta, varsigma_dn_d, ...
                                                       reshape(varsigma_dt(d, :, :), T, AO), varsigma_ao, gama, Omega{d}); 
           
       end
        
       fprintf('                                  use time : %d    \n', toc);
               
       % update a level
       fprintf('             update a level  \n');
       tic;
       
       for a = 1 : A
           
           d_a = find(AD(a, :) > 0);
           
           % udpate u_a and r_a 
           %fprintf('             update a level: u_%d and r_%d  \n', a, a); 
           [ua, ra] = update_ur_a(u_a(a, :), r_a(a, :), d_a, a, A, O, AD, varsigma_dt, alpha_a, step, nu_a_s, Omega);
           
           u_a(a, :) = ua;
           r_a(a, :) = ra;
           
           % udpate varsigma_ao
           %fprintf('             update a level: varsigma_ao for %d \n', a);
           varsigma_ao(a, :, :) = update_varsigma_ao(reshape(varsigma_ao(a, :, :), O, K), a, d_a, T, A, O, K, V, AD, ...
                                    dn_list, d_list, v_list, u_0, r_0, varsigma_dn, varsigma_dt, psi_vartheta, gama, step, Omega, z_a_s);
       end
       
       fprintf('                                  use time : %d  \n', toc);

       % sample nu_ao and z_ao
       fprintf('             sample nu_a and z_a  \n');
       tic;
       [nu_a_s, pi_a_s, z_a_s] = sample_a2(S, A, O, K, u_a, r_a, varsigma_ao);
       
       parfor d = 1 : D
           a_d       = find(AD(:, d) > 0);
           A_d       = Ad(d);
           Omega_d   = evaluate_omega(pi_a_s(:, a_d, :), z_a_s(:, a_d, :), A_d, O, S, K);
           Omega{d}  = Omega_d;
       end
       
       fprintf('                                  use time : %d  \n', toc);
              
       % update k level
       fprintf('             update k level  \n');
       tic;
       
       % udpate u_0 and r_0 
       %fprintf('             update k level: u_0 and r_0  \n');
       %tic;
       u_0        = update_u_0(u_0, K, varsigma_ao, gama);
       r_0        = update_r_0(r_0, K, varsigma_ao, alpha_0, gama);
       %fprintf('                                  use time : %d  \n', toc);
        
       % udpate vartheta
       %fprintf('             update k level: vartheta  \n');
       %tic;
       vartheta   = update_vartheta(vartheta, K, V, T, AO, dn_list, d_list, v_list, ...
                                    varsigma_dt, varsigma_ao, varsigma_dn, eta, gama);
       
       psi_vartheta  = psi(vartheta) - psi(repmat(sum(vartheta, 2), 1, V));
           
       fprintf('                                  use time : %d  \n', toc);
       
       % evaluate ELBO
       fprintf('             evaluate ELBO  \n');
       tic;
       L         = evaluate_likelihd(K, D, V, A, T, O, Ad, AO, AD, DN, Nd, d_list, v_list, u_0, r_0, u_a, r_a, u_d, r_d, ...
                             varsigma_dn, varsigma_dt, varsigma_ao, vartheta, psi_vartheta, alpha_0, alpha_a, alpha_d, eta, Omega, z_a_s );
       fprintf('                                  L = %d,  use time : %d  \n', L, toc);
       
       %% output       
       L_list(iter)       = L;
       
       CHDPMV.L           = L;
       CHDPMV.u_0         = u_0;
       CHDPMV.r_0         = r_0;
       CHDPMV.u_a         = u_a;
       CHDPMV.r_a         = r_a;
       CHDPMV.u_d         = u_d;
       CHDPMV.r_d         = r_d;
       CHDPMV.varsigma_ao = varsigma_ao;
       CHDPMV.varsigma_dt = varsigma_dt;
       CHDPMV.varsigma_dn = varsigma_dn;
       CHDPMV.vartheta    = vartheta;
              
       change = evaluate_variable_change(CHDPMV_old, CHDPMV)
                            
       change_list(iter)  = change;       
       
       if change < 0.01
          break; 
       end
              
       iter              = iter + 1;        
    end
    
end


function [Nd, Ad, AO, DN, S, dn_list, d_list, v_list, T, O, K, u_0, r_0, u_a, r_a, u_d, r_d, ...
    varsigma_ao, varsigma_dt, varsigma_dn, vartheta, gama, stepalpha] = initialization(A, D, V, AD, DV)

    % Max number of iteration
%     Max_iteration = 5000;
        
    % number of authors of douments
    Ad   =  sum(AD ~= 0, 1);
    
    % number of words of douments
    Nd   =  sum(DV ~= 0, 2);
    
    DN   = sum(Nd);
    
    % data list    
    dn_list = find(DV > 0);
    [d_list, v_list] = ind2sub([D V], dn_list);
        
    % proximal parameter
    gama = 0.1;
    
    % gradient step
    stepalpha = 0.2;
    
    % sample number
    S = 1000;
               
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% variables
    
    % truncations
    K        = D;
    O        = K;
    T        = K;    
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
    for k = 1 : K-1
        r_0(k) = ( alpha_0 - 1 + sum(sum(sum(varsigma_ao(:, :, k+1:K), 1), 2), 3) + gama*(r_0_i(k)-1) )/(1+gama) + 1;
    end
end

function vartheta  = update_vartheta(vartheta_i, K, V, T, AO, ...
    dn_list, d_list, v_list, varsigma_dt, varsigma_ao, varsigma_dn, eta, gama)

    vartheta = zeros(K, V);
    
    for k = 1 : K
        varsigma_aok = transpose(reshape(varsigma_ao(:, :, k), AO, 1));
        
        for v = 1 : V
            
            v_list_v  = find(v_list == v);
            d_list_v  = d_list(v_list_v);
            %dn_list_v = dn_list(v_list_v);
            DNV       = length(v_list_v);
            
            tmp = sum(sum(sum(reshape(repmat(varsigma_aok, DNV*T, 1), DNV, T, AO) ...
                    .* varsigma_dt(d_list_v, :, :)...
                    .* repmat(varsigma_dn(v_list_v, :), 1, 1, AO) ))) ;
            
            vartheta(k, v) = (eta + gama*vartheta_i(k, v) + tmp ) / (1+gama) ;
        end  
    end

end

%%%%%%%%%%%%%%%%%%%%%% the following is unique for CHDPM 

function [nu_a_s, pi_a_s, z_a_s] = sample_a(S, A, O, K, u_a, r_a, varsigma_ao)
    
    nu_a_s      = zeros(S, A, O);
    pi_a_s      = zeros(S, A, O);
    z_a_s       = zeros(S, A, O);
    
    for s = 1 : S
        nu_as           = betarnd(u_a, r_a);        
        tmp             = cumprod(1-nu_as, 2);
        tmp             = [ones(A, 1) tmp];
        nu_a_s(s, :, :) = [nu_as ones(A, 1)] ;
        pi_a_s(s, :, :) = [nu_as ones(A, 1)] .* tmp;
        z_as            = mnrnd(ones(A*O, 1), reshape(varsigma_ao, A*O, K));
        [ao_idx, k_idx] = find(z_as == 1);
        tmpp            = zeros(A, O);
        tmpp(ao_idx)    = k_idx;
        z_a_s(s, :, :)  = tmpp;
    end
end

function [nu_a_s, pi_a_s, z_a_s] = sample_a2(S, A, O, K, u_a, r_a, varsigma_ao)
    
    nu_a_s      = zeros(S, A, O);
    pi_a_s      = zeros(S, A, O);
    z_a_s       = zeros(S, A, O);
    
    for s = 1 : S
        nu_as           = u_a ./ (u_a+r_a);        
        tmp             = cumprod(1-nu_as, 2);
        tmp             = [ones(A, 1) tmp];
        nu_a_s(s, :, :) = [nu_as ones(A, 1)] ;
        pi_a_s(s, :, :) = [nu_as ones(A, 1)] .* tmp;
        z_as            = mnrnd(ones(A*O, 1), reshape(varsigma_ao, A*O, K));
        [ao_idx, k_idx] = find(z_as == 1);
        tmpp            = zeros(A, O);
        tmpp(ao_idx)    = k_idx;
        z_a_s(s, :, :)  = tmpp;
    end
end

function Omega_d = evaluate_omega(pi_a_s_d, z_a_s_d, A_d, O, S, K)

    Omega_d = zeros(S, A_d, O);
    
    for s = 1 : S
        
        z_asd  = reshape(z_a_s_d(s, :, :), A_d, O);
        pi_asd = reshape(pi_a_s_d(s, :, :), A_d, O);
        
        for k = 1 : K            
            pi_asd_tmp           = zeros(A_d, O);            
            idx_k                = find(z_asd == k); 
            
            if ~isempty(idx_k)
                pi_asd_tmp(idx_k)    = pi_asd(idx_k);            
                [~, ai]              = max(sum(pi_asd_tmp, 2)); 
                idx_k_ai             = find(z_asd(ai, :) == k);
                tmp                  = pi_asd(ai, idx_k_ai);
                pi_asd(idx_k)        = 0;
                pi_asd(ai, idx_k_ai) = tmp;
            end
        end
        
        Omega_d(s, :, :) = log(pi_asd/sum(sum(pi_asd)) + eps);
    end
    
end

function varsigma_dt_d = update_varsigma_dt(a_d, A_d, dn_v_list, dn_length, A, O, ...
    T, K, AO, psi_vartheta, varsigma_dn_d, varsigma_dt_i, varsigma_ao, gama, Omega_d)

    varsigma_dt_d    = zeros(T, AO);    
    
    Omegad_epc       = reshape(mean(Omega_d, 1), A_d, O);    
    psi_vartheta_tmp = transpose(psi_vartheta(:, dn_v_list));
    
    for t = 1 : T
        
        p       = ones(1, A_d*O);
        ao_idx  = ones(1, A_d*O);
        dt_tmp  = zeros(1, AO);
        
        varsigma_dn_dt_tmp = repmat(varsigma_dn_d(:, t), 1, K);
        
        for aoi = 1 : A_d*O
            
            [ai, o]     = ind2sub([A_d O], aoi);
            a           = a_d(ai);            
            ao          = sub2ind([A O], a, o);            
            ao_idx(aoi) = ao;
            
            tmp         = Omegad_epc(ai, o) - (1 + gama) + gama * log(varsigma_dt_i(t, ao)+eps)...
                          + sum(sum(repmat(reshape(varsigma_ao(a, o, :), 1, K), dn_length, 1)...
                            .* varsigma_dn_dt_tmp .* psi_vartheta_tmp ));
            
            p(aoi)      = tmp/(1+gama);
        end
        
        dt_tmp(ao_idx)      = 1 ./ sum(exp(repmat(p, [A_d*O 1]) - repmat(p', [1 A_d*O])), 2);        
        varsigma_dt_d(t, :) = dt_tmp;
    end
end

function [ua, ra] = update_ur_a(ua_i, ra_i, d_a, a, A, O, AD, varsigma_dt, alpha_a, step, nu_a_s, Omega)
    
    ua  = zeros(1, O-1);
    ra  = zeros(1, O-1);
    
    tmp  = (0 - psi(1, ua_i+ra_i)) .* (alpha_a - 1 - ua_i + 1 - ra_i + 1) ;
    tmpu = psi(1, ua_i) .* (1 - ua_i); 
    tmpr = psi(1, ra_i) .* (alpha_a - 1 - ra_i +1);
    
    psi_u_tmp = psi(ua_i + ra_i) - psi(ua_i);
    psi_r_tmp = psi(ua_i + ra_i) - psi(ra_i);
           
    for o = 1 : O-1
        
        ao   = sub2ind([A O], a, o);
        
        d_ua = tmp(o) + tmpu(o);        
        d_ra = tmp(o) + tmpr(o);
        
        for di = 1 : length(d_a)
            
            d       = d_a(di);            
            Omega_d = Omega{d};            
            ai      = sum(AD(1:a, d));
            
            d_ua = d_ua + sum(varsigma_dt(d, :, ao) * mean(Omega_d(:, ai, o) ...
                    .* (psi_u_tmp(o) + log(nu_a_s(:, a, o) + eps))));
            d_ra = d_ra + sum(varsigma_dt(d, :, ao) * mean(Omega_d(:, ai, o) ...
                    .* (psi_r_tmp(o) + log(1-nu_a_s(:, a, o) + eps))));
        end
        
        ua(o)  = ua_i(o) + d_ua*step;        
        stepua = step;        
        
        while ua(o) < 0 %|| evaluate_likelihd_u_a(ua_i(o), ua(o), ra_i(o), alpha_a ) == 0
            stepua = stepua*0.1;
            ua(o)  = ua_i(o) + d_ua*stepua;
        end
        
        ra(o)  = ra_i(o) + d_ra*step;
        stepra = step;
        
        while ra(o) < 0 %|| evaluate_likelihd_r_a(ra_i(o), ra(o), ua_i(o), alpha_a ) == 0
            stepra = stepra*0.1;
            ra(o)  = ra_i(o) + d_ra*stepra;
        end
    end

end

function varsigma_ao_a = update_varsigma_ao(varsigma_ao_a_i, a, d_a, T, A, O, K, V, AD, ...
    dn_list, d_list, v_list, u_0, r_0, varsigma_dn, varsigma_dt, psi_vartheta, gama, step, Omega, z_a_s)

    varsigma_ao_a = zeros(O, K);    
    d_list_a      = ismember(d_list, d_a);
    %dn_list_a     = dn_list(d_list_a);
    v_list_a      = v_list(d_list_a);  
    d_list_a      = d_list(d_list_a);        
    DNA           = length(d_list_a);
        
    for o = 1 : O
        
        ao = sub2ind([A O], a, o);        
        p  = zeros(1, K);
        
        for k = 1 : K
            
            d_varsigma_aok  = sum(sum(reshape(varsigma_dt(d_list_a, :, ao), DNA, T) .* varsigma_dn(d_list_a, :) .* repmat(transpose(psi_vartheta(k, v_list_a)), 1, T)));
            
            if k < K
                d_varsigma_aok  = d_varsigma_aok + psi(u_0(k)) - psi(u_0(k) + r_0(k)) ;
            end
            
            d_varsigma_aok  = d_varsigma_aok + sum(psi(r_0(1:k-1)) - psi(u_0(1:k-1) + r_0(1:k-1)))...
                                - log(varsigma_ao_a_i(o, k)+eps) - (1 + gama);
            
            for di = 1 : length(d_a)
                
                d              = d_a(di);
                Omega_d        = Omega{d};
                ai             = sum(AD(1:a, d));
                d_varsigma_aok = d_varsigma_aok + sum(varsigma_dt(d, :, ao) ...
                                    * mean(Omega_d(:, ai, o) .* (z_a_s(:, a, o)==k)) / varsigma_ao_a_i(o, k));
            end
            
            varsigma_ao_a_k = varsigma_ao_a_i(o, k) + d_varsigma_aok*step; % revise           
            steptmp         = step;
            
            while varsigma_ao_a_k < 0
                steptmp         = steptmp*0.1;
                varsigma_ao_a_k = varsigma_ao_a_i(o, k) + d_varsigma_aok*steptmp;
            end
            
            p(k) = varsigma_ao_a_k;
        end
        
        varsigma_ao_a(o, :)   = p ./ sum(p); 
    end
end

function L = evaluate_likelihd(K, D, V, A, T, O, Ad, AO, AD, DN, Nd, d_list, v_list, u_0, r_0, u_a, r_a, u_d, r_d, ...
    varsigma_dn, varsigma_dt, varsigma_ao, vartheta, vartheta_tmp, alpha_0, alpha_a, alpha_d, eta, Omega, z_a_s )

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
    for d = 1 : D
       
        A_d     = Ad(d);
        a_d     = find(AD(:, d) > 0);
        Omega_d = Omega{d};
        
        for aio = 1 : A_d*O
           
            [ai, o] = ind2sub([A_d O], aio);            
            a       = a_d(ai);
            ao      = sub2ind([A O], a, o);
            tmp     = 0;
            
            for k = 1 : K
                tmp = tmp + varsigma_ao(a, o, k) * mean(Omega_d(:, ai, o) .* (z_a_s(:, a, o) == k))/varsigma_ao(a,o,k);
            end
            
            L = L + sum(varsigma_dt(d, :, ao) * tmp);
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

function keep = evaluate_likelihd_varsigma_dt_d(varsigma_dt_dt_old, varsigma_dt_dt_new, d, t, A_d, a_d, Omega_d, K, A, O, AO, d_list, v_list, ...
    varsigma_dn, varsigma_ao, z_a_s, psi_vartheta )

    L_dt_old = 0;
    L_dt_new = 0;
    
    % z_dt    
    for aio = 1 : A_d*O
        
        [ai, o] = ind2sub([A_d O], aio);
        a       = a_d(ai);
        ao      = sub2ind([A O], a, o);
        tmp     = 0;
        
        for k = 1 : K
            tmp = tmp + varsigma_ao(a, o, k) * mean(Omega_d(:, ai, o) .* (z_a_s(:, a, o) == k))/varsigma_ao(a,o,k);
        end
        
        L_dt_old = L_dt_old + varsigma_dt_dt_old(ao) * tmp;
        L_dt_new = L_dt_new + varsigma_dt_dt_new(ao) * tmp;
    end
       
    L_dt_old = L_dt_old - sum(varsigma_dt_dt_old .* log(varsigma_dt_dt_old + eps) );
    L_dt_new = L_dt_new - sum(varsigma_dt_dt_new .* log(varsigma_dt_dt_new + eps) );
    
    % z_dn
    dn_list = find(d_list == d);
    
    for dn = 1 : length(dn_list)
        v = v_list(dn);
        L_dt_old = L_dt_old + sum(sum(reshape(varsigma_ao, AO, K) .* repmat(reshape(varsigma_dt_dt_old, AO, 1), 1, K) * varsigma_dn(dn, t) .* repmat(reshape(psi_vartheta(:, v), 1, K), AO, 1) )); 
        L_dt_new = L_dt_new + sum(sum(reshape(varsigma_ao, AO, K) .* repmat(reshape(varsigma_dt_dt_new, AO, 1), 1, K) * varsigma_dn(dn, t) .* repmat(reshape(psi_vartheta(:, v), 1, K), AO, 1) )); 
    end
    
    keep = L_dt_new > L_dt_old;
        
end

function keep = evaluate_likelihd_u_a(u_a_ao_old, u_a_ao_new, r_a_ao, alpha_a )
    
    % nu_a
    Lold = (alpha_a-1) * ( - psi(u_a_ao_old+r_a_ao)) ...
             - ( gammaln(u_a_ao_old+r_a_ao) - gammaln(u_a_ao_old)  + ...
             (u_a_ao_old-1) .* (psi(u_a_ao_old) - psi(u_a_ao_old+r_a_ao)) + (r_a_ao-1) .* ( - psi(u_a_ao_old+r_a_ao)) ); 
         
    Lnew = (alpha_a-1) * ( - psi(u_a_ao_new+r_a_ao)) ...
             - ( gammaln(u_a_ao_new+r_a_ao) - gammaln(u_a_ao_new)  + ...
             (u_a_ao_new-1) .* (psi(u_a_ao_new) - psi(u_a_ao_new+r_a_ao)) + (r_a_ao-1) .* ( - psi(u_a_ao_new+r_a_ao)) ); 
    
    keep = Lnew > Lold;
end

function keep = evaluate_likelihd_r_a(r_a_ao_old, r_a_ao_new, u_a_ao, alpha_a )
    
    % nu_a
    Lold =  (alpha_a-1) * (psi(r_a_ao_old) - psi(u_a_ao+r_a_ao_old)) ...
             - ( gammaln(u_a_ao+r_a_ao_old) - gammaln(u_a_ao) - gammaln(r_a_ao_old) + ...
             (u_a_ao-1) .* (psi(u_a_ao) - psi(u_a_ao+r_a_ao_old)) + (r_a_ao_old-1) .* (psi(r_a_ao_old) - psi(u_a_ao+r_a_ao_old)) ); 
         
    Lnew =  (alpha_a-1) * (psi(r_a_ao_new) - psi(u_a_ao+r_a_ao_new)) ...
             - ( gammaln(u_a_ao+r_a_ao_new) - gammaln(u_a_ao) - gammaln(r_a_ao_new) + ...
             (u_a_ao-1) .* (psi(u_a_ao) - psi(u_a_ao+r_a_ao_new)) + (r_a_ao_new-1) .* (psi(r_a_ao_new) - psi(u_a_ao+r_a_ao_new)) ); 
    
    keep = Lnew > Lold;
end

function change = evaluate_variable_change(CHDPMV_old, CHDPMV)

    change = mean(abs(CHDPMV_old.u_0 - CHDPMV.u_0)) + mean(abs(CHDPMV_old.r_0 - CHDPMV.r_0));
    change = change + mean(mean(abs(CHDPMV_old.u_a - CHDPMV.u_a))) + mean(mean(abs(CHDPMV_old.r_a - CHDPMV.r_a))); 
    change = change + mean(mean(abs(CHDPMV_old.u_d - CHDPMV.u_d))) + mean(mean(abs(CHDPMV_old.r_d - CHDPMV.r_d)));
    
    change = change + mean(mean(abs(CHDPMV_old.varsigma_dn - CHDPMV.varsigma_dn))) ;
    change = change + mean(mean(abs(CHDPMV_old.vartheta - CHDPMV.vartheta)));
    change = change + mean(mean(mean(abs(CHDPMV_old.varsigma_dt - CHDPMV.varsigma_dt))));
    change = change + mean(mean(mean(abs(CHDPMV_old.varsigma_ao - CHDPMV.varsigma_ao))));
    
    change = change / 10;
end


