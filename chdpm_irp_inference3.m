% 
% CHDP with maximization 
%    
%    using international restaurant process 
% 
%       by posterior inference with data
%
%          note: data is represented by d_list and v_list 
%               
%            for convergence evaluation. Output all the samples of K and L
%

function [ K_list, L_list, CHDPM ] = chdpm_irp_inference3( A, D, V, AD, Nd, dv_d_list, dv_v_list, alpha_0, alpha_a, alpha_d, gam, Max_iteration )
  
    K_list  = [];
    L_list  = [];
        
    % initialization
    [Ad, Sd, ad_cell, Td, Oa, K, phi, lambda, varphi, eta, theta, TdC, OaC, KC] = initialization(A, D, V, AD, Nd, dv_d_list, dv_v_list, gam);
        
    % output 
    lambda_out = lambda;
    varphi_out = varphi;
    eta_out    = eta;
    theta_out  = theta;
    K_out      = K;
    TdC_out    = TdC;
    OaC_out    = OaC;
    Lmax       = -Inf;
    
    % iteration
    iter   = 1;
    burnin = floor(Max_iteration * 0.2);
        
    %hbb=figure('name','irp-CHDP');
    
    while iter <= Max_iteration
        
       fprintf(' --------iteration num = %d \n', iter);
        
       % update phi
       fprintf('             update phi  \n');
       tic;
       [Td, Oa, K, phi, lambda, varphi, eta, theta, TdC, OaC, KC] = Update_phi(V, D, A, AD, dv_d_list, dv_v_list, Ad, Nd, Sd, ad_cell, ...
          Td, Oa, K, phi, lambda, varphi, eta, theta, TdC, OaC, KC, alpha_0, alpha_a, alpha_d, gam, iter);       
       fprintf('                                  use time : %d  \n', toc);

       % update varphi
       fprintf('             update varphi  \n');
       tic;
       [Oa, K, lambda, varphi, eta, theta, OaC, KC, vec_table] = Update_varphi(D, A, V, AD, dv_d_list, dv_v_list, Ad, Nd, Sd, ad_cell,...
          Td, Oa, K, phi, lambda, varphi, eta, theta, OaC, KC, alpha_0, alpha_a, gam, iter);       
       fprintf('                                  use time : %d  \n', toc);

       % update eta 
       fprintf('             update eta  \n');
       tic;
       [K, eta, theta, KC, vec_option] = Update_eta(A, V, dv_d_list, dv_v_list, vec_table, Oa, K, eta, KC, phi, lambda, varphi, theta, alpha_0, gam, iter);
       fprintf('                                  use time : %d  \n', toc);

       % update theta 
       fprintf('             update theta & L  \n');
       tic;
       [theta, L]  = Update_theta(K, D, V, dv_d_list, dv_v_list, vec_option, phi, lambda, varphi, eta, theta, gam);
       fprintf('                                  use time : %d  \n', toc);

%        % update alpha 
%        fprintf('             update alpha  \n');
%        tic;
%        [alpha_d]  = update_alphad(alpha_d, D, Td, Nd);
%        [alpha_a]  = update_alphaa(A, lambda,Oa, alpha_a);
%        [alpha_0]  = update_alpha0(K, Oa, alpha_0);
%        fprintf('                                                        alpha_0 = %d alpha_a = %d alpha_d = %d  \n', alpha_0, alpha_a, alpha_d);
%        fprintf('                                  use time : %d  \n', toc);

       %% output
       fprintf(' --------                                                  K  = %d    L = %d \n', K, L);
       fprintf(' --------       iteration = %d                             K  = %d  Oall = %d Omax = %d Tall = %d Tmax = %d  \n', iter, K, sum(Oa), max(Oa), sum(Td), max(Td));
       
       K_list(iter) = K;
       L_list(iter) = L;
       iter         = iter + 1;
       
       if iter > burnin && L > Lmax
          Lmax       = L;
          lambda_out = lambda;
          varphi_out = varphi;
          eta_out    = eta;
          theta_out  = theta;
          K_out      = K;
          TdC_out    = TdC;
          OaC_out    = OaC;
       end
       
       %% plot       
%        plot(K_list);
%        title('K_list')        
%        drawnow;
%        hold on;
              
    end
    
    CHDPM.K      = K_out;
    CHDPM.L      = Lmax;
    CHDPM.theta  = theta_out;
    CHDPM.eta    = eta_out;
    CHDPM.varphi = varphi_out;
    CHDPM.lambda = lambda_out;
    CHDPM.TdC    = TdC_out;
    CHDPM.OaC    = OaC_out;
    
end


function [Ad, Sd, ad_cell, Td, Oa, K, phi, lambda, varphi, eta, theta, TdC, OaC, KC] = initialization(A, D, V, AD, Nd, dv_d_list, dv_v_list, gam)

    % Max number of iteration
%     Max_iteration = 5000;
        
    % number of authors of douments
    Ad =  sum(AD ~= 0, 1);
    
    % number of words of douments
    %Nd =  sum(DV ~= 0, 2);
               
    % number of words of douments
    Sd = zeros(1, D);
    
    % author list of documents
    ad_cell = cell(1, D);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% latent variables
    
    % number of dishs
    K        = 1;

    % topics Dirichlet prior
    theta   = gamrnd(gam, 1, 1, V);
    theta   = theta / sum(theta);
    
    % number of options of authors
    Oa       = ones(A, 1);
      
    % eta(a, o) = k, i.e., dish k index of option o of author a
    eta      = ones(A, 1);
    
    % number of tables of douments
    Td       = zeros(D, 1);
    
    % varphi(d, t, :) = [a, o], i.e., chef a and option o index of table t of document d
    lambda   = zeros(D, 1);
    varphi   = zeros(D, 1);
        
    for a = 1 : A
       
        d_list     = find(AD(a, :) > 0);        
        Td(d_list) = Td(d_list) + 1;
        
        for di = 1 : length(d_list)
            d                = d_list(di);
            lambda(d, Td(d)) = a; 
            varphi(d, Td(d)) = 1;
        end        
    end
    
    % phi(d, v) = t, i.e., table t index of word v of document d    
    phi      = zeros(1, length(dv_d_list));
    
    Tdempty  = Td;
    
    for dn = 1 : length(dv_d_list)        
        d       = dv_d_list(dn); 
        
        if Tdempty(d) > 0
            phi(dn)    = Td(d) - Tdempty(d) + 1;
            Tdempty(d) = Tdempty(d) - 1;
        else
            phi(dn)    = find(mnrnd(1, 1/Td(d) * ones(1, Td(d))) == 1);
        end
                       
    end
            
    % TdC(d, t) = c, i.e., count of customers on table t of documents
    TdC = zeros(D, max(Td));
    tmp = 1;
    for d = 1 : D
       phi_d_idx = find(dv_d_list == d);
       for t = 1 : Td(d)
          phi_t_idx = find(phi == t);
          TdC(d,t)  = length(intersect(phi_d_idx, phi_t_idx));
       end
       Sd(d) = tmp;
       tmp   = tmp + Nd(d);
       
       ad_cell{d} = find(AD(:, d) > 0);
    end
          
    % OaC(a, 0) = c, i.e., count of tables with option o of author
    OaC = zeros(A, 1);
    
    for a = 1 : A
        idx_a     = find(lambda == a);        
        OaC(a, 1) = numel(idx_a);   
    end
     
    % KC(k) = c, i.e., count of dishes with k 
    KC = A;   
     
end


function [Td, Oa, K, phi, lambda, varphi, eta, theta, TdC, OaC, KC] = Update_phi(V, D, A, AD, dv_d_list, dv_v_list, Ad, Nd, Sd, ad_cell, ...
          Td, Oa, K, phi, lambda, varphi, eta, theta, TdC, OaC, KC, alpha_0, alpha_a, alpha_d, gam, iter)

    dold = -1;   
    %%                 
    for dn = 1 : length(dv_d_list)
        
        d   = dv_d_list(dn);
        v   = dv_v_list(dn);
        A_d = Ad(d);
        %a_d = find(AD(:,d) > 0);
        a_d = ad_cell{d};
        
        %% remove empty table of former d
        if d ~= dold && dold > 0
            idx_dt_empty_first = find(TdC(dold, 1 : Td(dold)) == 0, 1);
            
            if ~isempty(idx_dt_empty_first)
                
                phi_d_idx = find(dv_d_list == dold);
                
                num = idx_dt_empty_first;
                
                for t = num : Td(dold)
                    
                    if TdC(dold, t) > 0
                        phi_t_idx       = find(phi == t);
                        phi(intersect(phi_d_idx, phi_t_idx))     = num;
                        varphi(dold, num)  = varphi(dold, t);
                        lambda(dold, num)  = lambda(dold, t);
                        TdC(dold, num)     = TdC(dold, t);
                        num                = num + 1;
                    else
                        a_old             = lambda(dold, t);
                        o_old             = varphi(dold, t);
                        OaC(a_old, o_old) = OaC(a_old, o_old) - 1;
                    end
                end
                
                lambda(dold, num:end)  = 0;
                varphi(dold, num:end)  = 0;
                TdC(dold, num:end)     = 0;
                Td(dold)               = num - 1;
            end
            
            % remove empty columns of varphi
            Tmax = max(Td);
            if Tmax < size(varphi, 2)
                lambda(:, Tmax+1:end)  = [];
                varphi(:, Tmax+1:end)  = [];
                TdC(:, Tmax+1:end)     = [];
            end
        end
         
        dold = d;
        
        %% remove this customer
        t_old         = phi(dn);
        TdC(d, t_old) = TdC(d, t_old) - 1;
        
        %% select a table
        idxs  = select_table(d, v, Td(d), a_d, A_d, V, K, Oa,...
            TdC, OaC, KC, phi, lambda, varphi, eta, theta, alpha_d, alpha_a, alpha_0, gam);
        
        switch idxs(1)
            
            % sit on an old table
            case 1
                t_s                      = idxs(2);
                phi(dn)                  = t_s;
                TdC(d, t_s)              = TdC(d, t_s) + 1;
                
            % sit on an new table but old option	  
            case 2
                t_s                      = Td(d) + 1;
                phi(dn)                  = t_s;
                Td(d)                    = Td(d) + 1;
                TdC(d, t_s)              = 1;
                a_s                      = idxs(2);
                o_s                      = idxs(3);
                lambda(d, t_s)           = a_s;
                varphi(d, t_s)           = o_s;
                OaC(a_s, o_s)            = OaC(a_s, o_s) + 1;
                
            % sit on a new table, new option, but old dish
            case 3
                t_s                      = Td(d) + 1;
                phi(dn)                  = t_s;
                Td(d)                    = Td(d) + 1;
                TdC(d, t_s)              = 1;
                a_s                      = idxs(2);
                o_s                      = Oa(a_s) + 1;
                Oa(a_s)                  = Oa(a_s) + 1;
                lambda(d, t_s)           = a_s;
                varphi(d, t_s)           = o_s;
                OaC(a_s, o_s)            = 1;
                k_s                      = idxs(3);                
                eta(a_s, o_s)            = k_s;
                KC(k_s)                  = KC(k_s) + 1;
                
                % sit on a new table, new option, new dish
            case 4
                t_s                      = Td(d) + 1;
                phi(dn)                  = t_s;
                Td(d)                    = Td(d) + 1;
                TdC(d, t_s)              = 1;
                a_s                      = idxs(2);
                o_s                      = Oa(a_s) + 1;
                Oa(a_s)                  = Oa(a_s) + 1;
                lambda(d, t_s)           = a_s;
                varphi(d, t_s)           = o_s;
                OaC(a_s, o_s)            = 1;
                k_s                      = K+1;               
                eta(a_s, o_s)            = k_s;
                K                        = K + 1;
                KC(K)                    = 1;
                theta_new                = gamrnd(gam, 1, 1, V);
                theta(K, :)              = theta_new / sum(theta_new);
        end
        
    end
    
    %% remove empty table if exist
    d = D;
    idx_dt_empty_first = find(TdC(d, 1 : Td(d)) == 0, 1);
    
    if ~isempty(idx_dt_empty_first)
        
        phi_d_idx = find(dv_d_list == d);
        
        num = idx_dt_empty_first;
        
        for t = num : Td(d)
            
            if TdC(d, t) > 0
                phi_t_idx       = find(phi == t);
                phi(intersect(phi_d_idx, phi_t_idx))     = num;
                varphi(d, num)  = varphi(d, t);
                lambda(d, num)  = lambda(d, t);
                TdC(d, num)     = TdC(d, t);
                num             = num + 1;
            else
                a_old             = lambda(d, t);
                o_old             = varphi(d, t);
                OaC(a_old, o_old) = OaC(a_old, o_old) - 1;
            end
        end
        
        lambda(d, num:end)  = 0;
        varphi(d, num:end)  = 0;
        TdC(d, num:end)     = 0;
        Td(d)               = num - 1;
    end
    
    % remove empty columns of varphi
    Tmax = max(Td);
    if Tmax < size(varphi, 2)
        lambda(:, Tmax+1:end)  = [];
        varphi(:, Tmax+1:end)  = [];
        TdC(:, Tmax+1:end)     = [];
    end
    
    %% remove empty option if exist
    for a = 1 : A

        idx_ao_empty_first = find(OaC(a, 1 : Oa(a)) == 0, 1);

        if ~isempty(idx_ao_empty_first)

            num     = idx_ao_empty_first;
            idx_a   = find(lambda == a);

            for o = num : Oa(a)

                if OaC(a, o) > 0
                    idx_o          = find(varphi == o);
                    idx_ao         = intersect(idx_a, idx_o);
                    varphi(idx_ao) = num;
                    eta(a, num)    = eta(a, o);
                    OaC(a, num)    = OaC(a, o);
                    num            = num + 1;
                else
                    k_old          = eta(a, o);
                    KC(k_old)      = KC(k_old) - 1;
                end
            end

            eta(a, num:end) = 0;
            OaC(a, num:end) = 0;
            Oa(a)           = num - 1;
        end
    end

    % remove empty columns of eta
    Omax                = max(Oa);
    if Omax < size(eta, 2)
        eta(:, Omax+1:end)  = [];
        OaC(:, Omax+1:end)  = [];
    end

    %% remove empty topic if exist
    idx_k_empty_first = find(KC(1:K) == 0, 1);

    if ~isempty(idx_k_empty_first)

        num = idx_k_empty_first;

        for k = num : K

            if KC(k) > 0
                idx           = find(eta == k);
                eta(idx)      = num;
                theta(num, :) = theta(k, :);
                KC(num)       = KC(k);
                num           = num + 1;
            end
        end

        % remove empty columns of K
        K                 = num - 1;
        theta(K+1:end, :) = [];
        KC(K+1:end)       = [];
    end
    
end


function [Oa, K, lambda, varphi, eta, theta, OaC, KC, vec_table] = Update_varphi(D, A, V, AD, dv_d_list, dv_v_list, Ad, Nd, Sd, ad_cell, ...
          Td, Oa, K, phi, lambda, varphi, eta, theta, OaC, KC, alpha_0, alpha_a, gam, iter)
       
    vec_table = cell(1, D);
      
    for d = 1 : D
       
        %a_d = find(AD(:, d) > 0);
        a_d = ad_cell{d};
        A_d = Ad(d);
        N_d = Nd(d);
        S_d = Sd(d);
        Td_d = Td(d);
        
        v_list_d_tmp    = dv_v_list(S_d:(S_d+N_d-1));
        phi_d_tmp       = phi(S_d:(S_d+N_d-1));
        
        vec_table_d = cell(1, Td_d);
        
        for t = 1 : Td_d
           
            %% remove this table
            ao_old                     = zeros(1, 2);
            ao_old(1)                  = lambda(d, t);
            ao_old(2)                  = varphi(d, t);            
            OaC(ao_old(1), ao_old(2))  = OaC(ao_old(1), ao_old(2)) - 1;
            
            %% sample an option  
            data_vec                 = zeros(1, V);
            phi_t_idx                = phi_d_tmp == t;
            v_idx                    = v_list_d_tmp(phi_t_idx);
            
            if length(v_idx) > 1
                tab                  = tabulate(v_idx);
                data_vec(tab(:, 1))  = tab(:, 2);
            else
                data_vec(v_idx)      = 1;
            end
            
            vec_table_d{t} = data_vec;
            
            idxs = select_option(data_vec, a_d, A_d, V, OaC, KC,...
                                            Oa, K, eta, theta, alpha_a, alpha_0, gam);
            
            switch idxs(1)
                
                % choose an old option
                case 1
                    a_s                      = idxs(2);
                    o_s                      = idxs(3);
                    lambda(d, t)             = a_s;
                    varphi(d, t)             = o_s;
                    OaC(a_s, o_s)            = OaC(a_s, o_s) + 1;
                    
                 % choose a new option, but old dish	
                case 2
                    a_s                      = idxs(2);
                    o_s                      = Oa(a_s) + 1;
                    Oa(a_s)                  = Oa(a_s) + 1;
                    lambda(d, t)             = a_s;
                    varphi(d, t)             = o_s;
                    OaC(a_s, o_s)            = 1;
                    k_s                      = idxs(3);
                    eta(a_s, o_s)            = k_s;
                    KC(k_s)                  = KC(k_s) + 1;
                    
                % choose a new option, new dish	    
                case 3
                    a_s                      = idxs(2);
                    o_s                      = Oa(a_s) + 1;
                    Oa(a_s)                  = Oa(a_s) + 1;
                    lambda(d, t)             = a_s;
                    varphi(d, t)             = o_s;
                    OaC(a_s, o_s)            = 1;
                    K                        = K + 1;
                    eta(a_s, o_s)            = K;
                    KC(K)                    = 1;
                    theta_new                = gamrnd(gam, 1, 1, V);
                    theta(K, :)              = theta_new / sum(theta_new);
            end
                          
        end
        
        vec_table{d} = vec_table_d;
        
        if rem(d, 50) == 0
            
            %% remove empty option if exist
            for a = 1 : A
                
                idx_ao_empty_first = find(OaC(a, 1 : Oa(a)) == 0, 1);
                
                if ~isempty(idx_ao_empty_first)
                    
                    num     = idx_ao_empty_first;
                    idx_a   = find(lambda == a);
                    
                    for o = num : Oa(a)
                        
                        if OaC(a, o) > 0
                            idx_o          = find(varphi == o);
                            idx_ao         = intersect(idx_a, idx_o);
                            varphi(idx_ao) = num;
                            eta(a, num)    = eta(a, o);
                            OaC(a, num)    = OaC(a, o);
                            num            = num + 1;
                        else
                            k_old          = eta(a, o);
                            KC(k_old)      = KC(k_old) - 1;
                        end
                    end
                    
                    eta(a, num:end) = 0;
                    OaC(a, num:end) = 0;
                    Oa(a)           = num - 1;
                end
            end
            
            % remove empty columns of eta
            Omax = max(Oa);
            if Omax < size(eta, 2)
                eta(:, Omax+1:end)  = [];
                OaC(:, Omax+1:end)  = [];
            end
        end
        
    end
      
    %% remove empty option if exist  
    for a = 1 : A
                
        idx_ao_empty_first = find(OaC(a, 1 : Oa(a)) == 0, 1);
        
        if ~isempty(idx_ao_empty_first)
            
            num     = idx_ao_empty_first;        
            idx_a   = find(lambda == a);

            for o = num : Oa(a)
                
                if OaC(a, o) > 0      
                    idx_o          = find(varphi == o);
                    idx_ao         = intersect(idx_a, idx_o);
                    varphi(idx_ao) = num;
                    eta(a, num)    = eta(a, o);
                    OaC(a, num)    = OaC(a, o);
                    num            = num + 1;
                else
                    k_old          = eta(a, o);
                    KC(k_old)      = KC(k_old) - 1;
                end
            end

            eta(a, num:end) = 0;
            OaC(a, num:end) = 0;
            Oa(a)           = num - 1; 
        end
    end  
         
    % remove empty columns of eta
    Omax = max(Oa); 
    if Omax < size(eta, 2)
        eta(:, Omax+1:end)  = []; 
        OaC(:, Omax+1:end)  = [];
    end
    
    %% remove empty topic if exist    
    idx_k_empty_first = find(KC(1:K) == 0, 1);
        
    if ~isempty(idx_k_empty_first)

        num = idx_k_empty_first;

        for k = num : K
            
            if KC(k) > 0
                idx           = find(eta == k);
                eta(idx)      = num;
                theta(num, :) = theta(k, :);
                KC(num)       = KC(k);
                num           = num + 1;
            end
        end

        % remove empty columns of K
        K                 = num - 1;
        theta(K+1:end, :) = [];
        KC(K+1:end)       = [];
    end  
   
end


function [K, eta, theta, KC, vec_option] = Update_eta(A, V, dv_d_list, dv_v_list, vec_table, Oa, K, eta, KC, phi, lambda, varphi, theta, alpha_0, gam, iter)
    
    vec_option = cell(1, A);

    for a = 1 : A

        O_a          = Oa(a);        
        idx_a        = find(lambda == a);
        vec_option_a = cell(1, O_a);
        
        for o = 1 : O_a
                        
            %% remove this option
            k_old      = eta(a, o);            
            KC(k_old)  = KC(k_old) - 1;

            %% select dish      
            data_vec         = zeros(1, V);
            idx_o            = find(varphi == o);
            idx_ao           = intersect(idx_a, idx_o);
            [d_list, t_list] = ind2sub(size(varphi), idx_ao);
            
            for i = 1 : length(d_list)  
%                 phi_d_idx       = find(dv_d_list == d_list(i));
%                 phi_t_idx       = find(phi == t_list(i));
%                 v_idx           = dv_v_list(intersect(phi_d_idx, phi_t_idx));
%                 if length(v_idx) > 1
%                     tab                 = tabulate(v_idx);
%                     data_vec(tab(:, 1)) = data_vec(tab(:, 1)) + transpose(tab(:, 2));
%                 else
%                     data_vec(v_idx)     = data_vec(v_idx) + 1;
%                 end

                data_vec = data_vec + vec_table{d_list(i)}{t_list(i)};

            end
            
            vec_option_a{o} = data_vec;
            
            [k_selected]    = select_dish(data_vec, KC, K, V, theta, alpha_0, gam);
            eta(a, o)       = k_selected;
            
            % if order a new dish
            if k_selected > K
                K             = K + 1;
                KC(K)         = 1;
                theta_new     = gamrnd(gam, 1, 1, V);
                theta(K, :)   = theta_new / sum(theta_new);
            else
                KC(k_selected) = KC(k_selected) + 1;
            end            
        end
        
        vec_option{a} = vec_option_a;
        
        if rem(a, 50) == 0
            
            %% remove empty topic if exist
            idx_k_empty_first = find(KC(1:K) == 0, 1);
            
            if ~isempty(idx_k_empty_first)
                
                num = idx_k_empty_first;
                
                for k = num : K
                    
                    if KC(k) > 0
                        idx           = find(eta == k);
                        eta(idx)      = num;
                        KC(num)       = KC(k);
                        theta(num, :) = theta(k, :);
                        num           = num + 1;
                    end
                end
                
                % remove empty columns of K
                K                 = num - 1;
                theta(K+1:end, :) = [];
                KC(K+1:end)       = [];
            end
            
        end
        
    end
    
    %% remove empty topic if exist
    idx_k_empty_first = find(KC(1:K) == 0, 1);
        
    if ~isempty(idx_k_empty_first)

        num = idx_k_empty_first;

        for k = num : K
            
            if KC(k) > 0
                idx           = find(eta == k);
                eta(idx)      = num;
                KC(num)       = KC(k);                
                theta(num, :) = theta(k, :);
                num           = num + 1;
            end
        end

        % remove empty columns of K
        K                 = num - 1;
        theta(K+1:end, :) = [];
        KC(K+1:end)       = [];
    end  
    
end


function [theta, L]  = Update_theta(K, D, V, dv_d_list, dv_v_list, vec_option, phi, lambda, varphi, eta, theta, gam)

    L = 0;
    
    for k = 1 : K
       
        data_vec         = zeros(1, V);
        idx_k            = find(eta == k);
        [idx_ka, idx_ko] = ind2sub(size(eta), idx_k);
        
        for i = 1 : length(idx_k)
%             idx_a            = find(lambda == idx_ka(i));
%             idx_o            = find(varphi == idx_ko(i));
%             idx_ao           = intersect(idx_a, idx_o);
%             [d_list, t_list] = ind2sub(size(lambda), idx_ao);
%                         
%             for j = 1 : length(d_list)  
%                 phi_d_idx       = find(dv_d_list == d_list(j));
%                 phi_v_idx       = find(phi == t_list(j));
%                 v_idx           = dv_v_list(intersect(phi_d_idx, phi_v_idx));
%                 if length(v_idx) > 1
%                     tab                 = tabulate(v_idx);
%                     data_vec(tab(:, 1)) = data_vec(tab(:, 1)) + transpose(tab(:, 2));
%                 else
%                     data_vec(v_idx)     = data_vec(v_idx) + 1;
%                 end                
%             end
            data_vec = data_vec + vec_option{idx_ka(i)}{idx_ko(i)};
        end
        
        theta_k     = gamrnd(gam + data_vec, 1);
        theta(k, :) = theta_k / sum(theta_k); 
        L           = L + sum(data_vec .* log(theta(k, :) + eps));
    end

end


function t_selected = select_table(d, v, T_d, a_d, A_d, V, K, Oa, ...
    TdC, OaC, KC, phi, lambda, varphi, eta, theta, alpha_d, alpha_a, alpha_0, gam)

    Oaall = sum(Oa(a_d));
    p_w   = zeros(1, T_d+Oaall+K+1);  
    p_L   = zeros(1, T_d+Oaall+K+1);
    p_a   = zeros(1, A_d);
    N_d   = sum(TdC(d, :));
    KCsum = sum(KC);
    num   = 1;
    k_weight_tmp = 0;
    
    k_t  = eta(sub2ind(size(eta), lambda(d, 1:T_d), varphi(d, 1:T_d)));
    p_w(num:(num+T_d-1))   = TdC(d, 1:T_d)/(N_d + alpha_d);
    p_L(num:(num+T_d-1))   = log(theta(k_t, v) + eps);  
    num                    = num + T_d;
    
    %
    p_w_a_tmp = zeros(1, Oaall);
    p_L_a_tmp = zeros(1, Oaall);
    numtmp = 1;
    k_list = zeros(1, Oaall);    
    a_list = zeros(1, Oaall);
    
    k_max_value = zeros(1, K);
    k_max_a     = zeros(1, K);
    O_a_min     = 100000000000;
    a_Oamin     = -1;
    
    for aidx = 1 : A_d
        
        a    = a_d(aidx);
        O_a  = sum(OaC(a, :));
        Oa_a = Oa(a);
        
        if Oa_a < O_a_min
           O_a_min = Oa_a; 
           a_Oamin = a;
        end
        
        k_max_value_a            = zeros(1, K);
        k_list_a                 = eta(a, 1:Oa_a);
        k_list(numtmp:(numtmp+Oa_a-1))    = k_list_a;
        a_list(numtmp:(numtmp+Oa_a-1))    = a;
           
        p_w_a_tmp(numtmp:(numtmp+Oa_a-1)) = OaC(a, 1:Oa_a) / (O_a+alpha_a);        
        p_L_a_tmp(numtmp:(numtmp+Oa_a-1)) = log(theta(k_list_a, v) + eps);
        
        for o = 1 : Oa_a
            k_max_value_a(k_list_a(o)) = k_max_value_a(k_list_a(o)) + p_w_a_tmp(numtmp+o-1);
        end
        
        %k_max_value_a(k_list_a) = k_max_value_a(k_list_a) + p_w(num:(num+Oa_a-1));
        
        idx_large               = find(k_max_value_a >= k_max_value);
        if ~isempty(idx_large)
            k_max_value(idx_large) = k_max_value_a(idx_large);
            k_max_a(idx_large)     = a;
        end
         
        numtmp                   = numtmp + Oa_a;
    end

    for k = 1 : K       
        a_k          = k_max_a(k);
        idx0         = find(k_list == k);
        idxna        = find(a_list ~= a_k);
        idx_0na      = intersect(idx0, idxna);
        p_w_a_tmp(idx_0na) = 0;        
    end
    
    p_w_sum            = sum(p_w_a_tmp) + alpha_a / (O_a_min+alpha_a);
    p_w_a_tmp          = p_w_a_tmp / p_w_sum;
    
    p_w(num:(num+Oaall-1)) = (alpha_d/(N_d + alpha_d)) * p_w_a_tmp;
    p_L(num:(num+Oaall-1)) = p_L_a_tmp;
    num                    = num+Oaall;
    
    %
    p_w(num:(num+K-1)) = (alpha_d/(N_d + alpha_d)) * (alpha_a / (O_a_min+alpha_a) / p_w_sum) * (KC / ( KCsum+alpha_0 ) );
    p_L(num:(num+K-1)) = log(theta(1:K, v) + eps);
    num                = num + K;
       
    p_w(num)           = (alpha_d/(N_d + alpha_d)) * (alpha_a / (O_a_min+alpha_a) / p_w_sum) * (alpha_0 / ( KCsum+alpha_0 ));
    p_L(num)           = -log(V + eps);
    
    p_w                = exp(p_L - max(p_L)) .* (p_w/sum(p_w)) ;    
    t_s                = find(mnrnd(1, p_w/sum(p_w)) == 1);
    
    if t_s <= T_d        
        t_selected = [1 t_s];
        return;
    else         
        t_s = t_s - T_d;
        for aidx = 1 : A_d
        
            a    = a_d(aidx);
            if t_s <= Oa(a)
                t_selected = [2 a, t_s];
                return;
            end
            t_s = t_s - Oa(a);
        end
        
        if t_s == K+1
            t_selected = [4 a_Oamin];
            return;
        else
            t_selected = [3 k_max_a(t_s) t_s];
            return;
        end
    end
end


function ao_selected = select_option(data_vec, a_d, A_d, V, OaC, KC,...
    Oa, K, eta, theta, alpha_a, alpha_0, gam)

    Oall   = sum(Oa(a_d));
    p_w    = zeros(1, Oall + (K+1));
    p_L    = zeros(1, Oall + (K+1));
    k_list = zeros(1, Oall);    
    a_list = zeros(1, Oall);
    
    KCsum  = sum(KC);
    Lik    = zeros(1, K+1);
    num    = 1;
    
    Lik(1:K) = sum(repmat(data_vec, K, 1) .* log(theta + eps), 2);
    Lik(K+1) = - sum(data_vec) * log(V);
    
    k_max_value = zeros(1, K);
    k_max_a     = zeros(1, K);
    O_a_min     = 100000000000;
    a_Oamin     = -1;
    
    for aidx = 1 : A_d
        
        a    = a_d(aidx);
        O_a  = sum(OaC(a, :));
        Oa_a = Oa(a);
        
        if Oa_a < O_a_min
           O_a_min = Oa_a; 
           a_Oamin = a;
        end
        
        k_max_value_a            = zeros(1, K);
        k_list_a                 = eta(a, 1:Oa_a);
        k_list(num:(num+Oa_a-1)) = k_list_a;
        a_list(num:(num+Oa_a-1)) = a;
        p_w(num:(num+Oa_a-1))    = OaC(a, 1:Oa_a) / (O_a+alpha_a);
        p_L(num:(num+Oa_a-1))    = Lik(k_list_a); 
        
        for o = 1 : Oa_a
            k_max_value_a(k_list_a(o)) = k_max_value_a(k_list_a(o)) + p_w(num+o-1);
        end
        
        %k_max_value_a(k_list_a) = k_max_value_a(k_list_a) + p_w(num:(num+Oa_a-1));
        
        idx_large               = find(k_max_value_a >= k_max_value);
        if ~isempty(idx_large)
            k_max_value(idx_large) = k_max_value_a(idx_large);
            k_max_a(idx_large)     = a;
        end
        
        num                   = num + Oa_a;
    end
    
    for k = 1 : K       
        a_k          = k_max_a(k);
        idx0         = find(k_list == k);
        idxna        = find(a_list ~= a_k);
        idx_0na      = intersect(idx0, idxna);
        p_w(idx_0na) = 0;        
    end
    
    p_w_sum            = sum(p_w) + alpha_a / (O_a_min+alpha_a);
    p_w                = p_w / p_w_sum;
    
    p_w(num:(num+K-1)) = alpha_a / (O_a_min+alpha_a) / p_w_sum * (KC / (KCsum + alpha_0));
    p_L(num:(num+K-1)) = Lik(1:K); 
    num                = num + K;    
    p_w(num)           = alpha_a / (O_a_min+alpha_a) / p_w_sum * (alpha_0 / (KCsum + alpha_0));
    p_L(num)           = Lik(K+1); 
    
    p                  = exp(p_L - max(p_L)) .* (p_w/sum(p_w));
    idx_ao             = find(mnrnd(1, p/sum(p)) == 1);
            
    for aidx = 1 : A_d
        
        a    = a_d(aidx);
        
        if idx_ao <= Oa(a)
            ao_selected = [1 a idx_ao];
            return;
        end
        idx_ao = idx_ao - Oa(a);                
    end
     
    if idx_ao == K+1
        ao_selected = [3 a_Oamin];
        return;
    else
        ao_selected = [2 k_max_a(idx_ao) idx_ao];
        return;
    end
    
end


function [k_selected] = select_dish(data_vec, KC, K, V, theta, alpha_0, gam)
    p            =  log(KC+eps) + transpose(sum(repmat(data_vec, [K 1]) .* log(theta +eps), 2)) ;    
    p            =  [p (log(alpha_0) - sum(data_vec) *log(V))];    
    p            = exp(p - max(p));
    k_selected   =  find(mnrnd(1, p/sum(p)) == 1);
end


function [alpha_d]  = update_alphad(alpha_d, D, Td, Nd)

    m = zeros(1, D);    
    w = zeros(1, D);
    s = zeros(1, D);

    for d = 1 : D
       
        m(d) = Td(d);        
        n    = Nd(d);        
        w(d) = betarnd(alpha_d + 1, n); 
        s(d) = binornd(1, n/(n+alpha_d));
        
    end

    alpha_d = gamrnd(1 + sum(m) - sum(s), 1/(1 - sum(log(w+eps)) ) );

end

function [alpha_a]  = update_alphaa(A, lambda,Oa, alpha_a)

    m = zeros(1, A);    
    w = zeros(1, A);
    s = zeros(1, A);

    parfor a = 1 : A
       
        m(a) = Oa(a);
        idx  = find(lambda == a)
        
        if isempty(idx)           
           w(a) = 1; 
           s(a) = 0;
        else
           n    = length(idx);
           w(a) = betarnd(alpha_a + 1, n); 
           s(a) = binornd(1, n/(n+alpha_a));
        end
        
    end

    alpha_a = gamrnd(1 + sum(m) - sum(s), 1/(1 - sum(log(w+eps)) ) );

end

function [alpha_0]  = update_alpha0(K, Oa, alpha_0)

    m       = K;
    n       = sum(Oa);

    w       = betarnd(alpha_0 + 1, n);    
    s       = binornd(1, n/(n+alpha_0));
    
    alpha_0 = gamrnd(1 + m - s, 1/(1 - log(w+eps)));

end
