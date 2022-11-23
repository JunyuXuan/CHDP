% 
% CHDP with superposition 
%    
%    using international restaurant process 
% 
%       by posterior inference with data
%
%          note: data is represented by d_list and v_list 
%


function [ K_list, L_list, CHDPS ] = chdps_irp_inference1( A, D, V, AD, Nd, dv_d_list, dv_v_list, alpha_0, alpha_a, alpha_d, gam, Max_iteration )
  
    K_list  = [];
    L_list  = [];
        
    % initialization
    [Ad, Td, Oa, K, phi, lambda, varphi, eta, theta, TdC, OaC, KC] = initialization(A, D, V, AD, Nd, dv_d_list, dv_v_list, gam);
        
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
       [Td, Oa, K, phi, lambda, varphi, eta, theta, TdC, OaC, KC] = Update_phi(V, D, A, AD, dv_d_list, dv_v_list, Ad, ...
          Td, Oa, K, phi, lambda, varphi, eta, theta, TdC, OaC, KC, alpha_0, alpha_a, alpha_d, gam, iter);       
       fprintf('                                  use time : %d  \n', toc);

       % update varphi
       fprintf('             update varphi  \n');
       tic;
       [Oa, K, lambda, varphi, eta, theta, OaC, KC] = Update_varphi(D, A, V, AD, dv_d_list, dv_v_list, Ad, ...
          Td, Oa, K, phi, lambda, varphi, eta, theta, OaC, KC, alpha_0, alpha_a, gam, iter);       
       fprintf('                                  use time : %d  \n', toc);

       % update eta 
       fprintf('             update eta  \n');
       tic;
       [K, eta, theta, KC] = Update_eta(A, V, dv_d_list, dv_v_list, Oa, K, eta, KC, phi, lambda, varphi, theta, alpha_0, gam, iter);
       fprintf('                                  use time : %d  \n', toc);

       % update theta 
       fprintf('             update theta & L  \n');
       tic;
       [theta, L]  = Update_theta(K, D, V, dv_d_list, dv_v_list, phi, lambda, varphi, eta, theta, gam);
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
    
    CHDPS.K      = K_out;
    CHDPS.L      = Lmax;
    CHDPS.theta  = theta_out;
    CHDPS.eta    = eta_out;
    CHDPS.varphi = varphi_out;
    CHDPS.lambda = lambda_out;
    CHDPS.TdC    = TdC_out;
    CHDPS.OaC    = OaC_out;
    
end


function [Ad, Td, Oa, K, phi, lambda, varphi, eta, theta, TdC, OaC, KC] = initialization(A, D, V, AD, Nd, dv_d_list, dv_v_list, gam)

    % Max number of iteration
%     Max_iteration = 5000;
        
    % number of authors of douments
    Ad =  sum(AD ~= 0, 1);
    
    % number of words of douments
    %Nd =  sum(DV ~= 0, 2);
               
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
    
    for d = 1 : D
       phi_d_idx = find(dv_d_list == d);
       for t = 1 : Td(d)
          phi_t_idx = find(phi == t);
          TdC(d,t)  = length(intersect(phi_d_idx, phi_t_idx));
       end
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


function [Td, Oa, K, phi, lambda, varphi, eta, theta, TdC, OaC, KC] = Update_phi(V, D, A, AD, dv_d_list, dv_v_list, Ad, ...
          Td, Oa, K, phi, lambda, varphi, eta, theta, TdC, OaC, KC, alpha_0, alpha_a, alpha_d, gam, iter)

    %%                 
    for dn = 1 : length(dv_d_list)
        
        d   = dv_d_list(dn);
        v   = dv_v_list(dn);
        A_d = Ad(d);
        a_d = find(AD(:,d) > 0);
        
        %% remove this customer
        t_old         = phi(dn);
        TdC(d, t_old) = TdC(d, t_old) - 1;
        
        %% select a table
        t_selected         = select_table(d, v, Td(d), a_d, A_d, V, K, Oa,...
            TdC, OaC, KC, phi, lambda, varphi, eta, theta, alpha_d, alpha_a, alpha_0, gam);
        phi(dn)            = t_selected;
        
        % sit on a new table
        if t_selected > Td(d)
            
            Td(d)                    = Td(d) + 1;
            TdC(d, t_selected)       = 1;
            
            % select option
            data_vec                 = zeros(1, V);
            data_vec(v)              = 1;
            [a_selected, o_selected] = select_option(data_vec, a_d, A_d, V, OaC, KC,...
                Oa, K, eta, theta, alpha_a, alpha_0, gam);
            lambda(d, t_selected)    = a_selected;
            varphi(d, t_selected)    = o_selected;
            
            % choose a new option
            if o_selected > Oa(a_selected)
                
                Oa(a_selected)              = Oa(a_selected) + 1;
                OaC(a_selected, o_selected) = 1;
                
                % select dish
                [k_selected]                = select_dish(data_vec, KC, K, V, theta, alpha_0, gam);
                eta(a_selected, o_selected) = k_selected;
                
                % order a new dish
                if k_selected > K
                    K             = K + 1;
                    KC(K)         = 1;
                    theta_new     = gamrnd(gam, 1, 1, V);
                    theta(K, :)   = theta_new / sum(theta_new);
                else
                    KC(k_selected) = KC(k_selected) + 1;
                end
            else
                OaC(a_selected, o_selected) = OaC(a_selected, o_selected) + 1;
            end
        else
            TdC(d, t_selected) = TdC(d, t_selected) + 1;
        end
        
        %% remove empty table if exist    
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
        
    end
    
%     %% remove empty table if exist
%     for d = 1 : D
%         idx_dt_empty_first = find(TdC(d, 1 : Td(d)) == 0, 1);
%         
%         if ~isempty(idx_dt_empty_first)
%             
%             phi_d_idx = find(dv_d_list == d);
%             
%             num = idx_dt_empty_first;
%             
%             for t = num : Td(d)
%                 
%                 if TdC(d, t) > 0
%                     phi_t_idx       = find(phi == t);
%                     phi(intersect(phi_d_idx, phi_t_idx))     = num;
%                     varphi(d, num)  = varphi(d, t);
%                     lambda(d, num)  = lambda(d, t);
%                     TdC(d, num)     = TdC(d, t);
%                     num             = num + 1;
%                 else
%                     a_old             = lambda(d, t);
%                     o_old             = varphi(d, t);
%                     OaC(a_old, o_old) = OaC(a_old, o_old) - 1;
%                 end
%             end
%             
%             lambda(d, num:end)  = 0;
%             varphi(d, num:end)  = 0;
%             TdC(d, num:end)     = 0;
%             Td(d)               = num - 1;
%         end
%         
%     end

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


function [Oa, K, lambda, varphi, eta, theta, OaC, KC] = Update_varphi(D, A, V, AD, dv_d_list, dv_v_list, Ad, ...
          Td, Oa, K, phi, lambda, varphi, eta, theta, OaC, KC, alpha_0, alpha_a, gam, iter)
       
    for d = 1 : D
       
        a_d = find(AD(:, d) > 0);
        A_d = Ad(d);
        
        phi_d_idx = find(dv_d_list == d);
        
        for t = 1 : Td(d)
           
            %% remove this table
            ao_old                     = zeros(1, 2);
            ao_old(1)                  = lambda(d, t);
            ao_old(2)                  = varphi(d, t);            
            OaC(ao_old(1), ao_old(2))  = OaC(ao_old(1), ao_old(2)) - 1;
            
            %% sample an option  
            data_vec                 = zeros(1, V);
%             v_idx                    = find(phi(d, :) == t);
%             data_vec(v_idx)          = 1;
            phi_t_idx                = find(phi == t);
            v_idx                    = dv_v_list(intersect(phi_d_idx, phi_t_idx));
            
            if length(v_idx) > 1
                tab                  = tabulate(v_idx);
                data_vec(tab(:, 1))  = tab(:, 2);
            else
                data_vec(v_idx)      = 1;
            end
            
            [a_selected, o_selected] = select_option(data_vec, a_d, A_d, V, OaC, KC,...
                                            Oa, K, eta, theta, alpha_a, alpha_0, gam);
            lambda(d, t)             = a_selected;
            varphi(d, t)             = o_selected;
            
            % if choose a new option
            if o_selected > Oa(a_selected)
                
                Oa(a_selected)              = Oa(a_selected) + 1;
                OaC(a_selected, o_selected) = 1;
                
                % select dish
                [k_selected]                = select_dish(data_vec, KC, K, V, theta, alpha_0, gam);                
                eta(a_selected, o_selected) = k_selected;
                
                % if order a new dish
                if k_selected > K
                    K             = K + 1;
                    KC(K)         = 1;
                    theta_new     = gamrnd(gam, 1, 1, V);
                    theta(K, :)   = theta_new / sum(theta_new);
                else
                    KC(k_selected) = KC(k_selected) + 1;
                end
            else
                OaC(a_selected, o_selected) =  OaC(a_selected, o_selected) + 1;
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


function [K, eta, theta, KC] = Update_eta(A, V, dv_d_list, dv_v_list, Oa, K, eta, KC, phi, lambda, varphi, theta, alpha_0, gam, iter)
    
    for a = 1 : A

        O_a   = Oa(a);        
        idx_a = find(lambda == a);
        
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
                phi_d_idx       = find(dv_d_list == d_list(i));
                phi_v_idx       = find(phi == t_list(i));
                v_idx           = dv_v_list(intersect(phi_d_idx, phi_v_idx));
                if length(v_idx) > 1
                    tab                 = tabulate(v_idx);
                    data_vec(tab(:, 1)) = data_vec(tab(:, 1)) + transpose(tab(:, 2));
                else
                    data_vec(v_idx)     = data_vec(v_idx) + 1;
                end
            end
            
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


function [theta, L]  = Update_theta(K, D, V, dv_d_list, dv_v_list, phi, lambda, varphi, eta, theta, gam)

    L = 0;
    
    for k = 1 : K
       
        data_vec         = zeros(1, V);
        idx_k            = find(eta == k);
        [idx_ka, idx_ko] = ind2sub(size(eta), idx_k);
        
        for i = 1 : length(idx_k)
            idx_a            = find(lambda == idx_ka(i));
            idx_o            = find(varphi == idx_ko(i));
            idx_ao           = intersect(idx_a, idx_o);
            [d_list, t_list] = ind2sub(size(lambda), idx_ao);
                        
            for j = 1 : length(d_list)  
                phi_d_idx       = find(dv_d_list == d_list(j));
                phi_v_idx       = find(phi == t_list(j));
                v_idx           = dv_v_list(intersect(phi_d_idx, phi_v_idx));
                if length(v_idx) > 1
                    tab                 = tabulate(v_idx);
                    data_vec(tab(:, 1)) = data_vec(tab(:, 1)) + transpose(tab(:, 2));
                else
                    data_vec(v_idx)     = data_vec(v_idx) + 1;
                end                
            end
        end
        
        theta_k     = gamrnd(gam + data_vec, 1);
        theta(k, :) = theta_k / sum(theta_k); 
        L           = L + sum(data_vec .* log(theta(k, :) + eps));
    end

end


function t_selected = select_table(d, v, T_d, a_d, A_d, V, K, Oa, ...
    TdC, OaC, KC, phi, lambda, varphi, eta, theta, alpha_d, alpha_a, alpha_0, gam)

    p   = zeros(1, T_d+1);    
    N_d = sum(TdC(d, :));
    
    for t = 1 : T_d        
        k_t        = eta(lambda(d,t), varphi(d,t));
        p(t)       = log(TdC(d, t) +eps) - log(N_d+alpha_d+eps) + log(theta(k_t, v) + eps);        
    end

    p(T_d + 1)  =  log(alpha_d+eps) - log(N_d+alpha_d+eps) + log(Likehd_dv(v, a_d, A_d, V, K, Oa, OaC, KC, eta, theta, alpha_a, alpha_0, gam) + eps);
    p           =  1 ./ sum(exp(repmat(p, [T_d+1 1]) - repmat(p', [1 T_d+1])), 2);
    t_selected  =  find(mnrnd(1, p) == 1);
    
end


function [a_selected, o_selected] = select_option(data_vec, a_d, A_d, V, OaC, KC,...
    Oa, K, eta, theta, alpha_a, alpha_0, gam)

    Oall   = sum(Oa(a_d));
    p_list = zeros(1, Oall + A_d);
    a_list = zeros(1, Oall + A_d);
    o_list = zeros(1, Oall + A_d);
    
    num = 1;

    for aidx = 1 : A_d
        
        a    = a_d(aidx);
        O_a  = sum(OaC(a, :));
                
        for o = 1 : Oa(a)    

            p_list(num) = log(OaC(a, o)+eps) - log(O_a+alpha_a + eps) ...
                            + sum(data_vec .* log(theta(eta(a, o), :) + eps) );            
            a_list(num) = a;
            o_list(num) = o;
            num         = num + 1;
        end
        
        p_list(num) = log(alpha_a) - log(O_a+alpha_a+eps) + log(Likehd_k(data_vec, V, K, KC, theta, alpha_0, gam)+eps);
        a_list(num) = a;
        o_list(num) = Oa(a)+1;
        num         = num + 1;
    end

    p           =   1 ./ sum(exp(repmat(p_list, [(num-1) 1]) - repmat(p_list', [1 (num-1)])), 2);
    idx_ao      =   find(mnrnd(1, p) == 1);
    
    a_selected  =   a_list(idx_ao);
    o_selected  =   o_list(idx_ao);
end


function [k_selected] = select_dish(data_vec, KC, K, V, theta, alpha_0, gam)
    p            =  log(KC+eps) + transpose(sum(repmat(data_vec, [K 1]) .* log(theta +eps), 2)) ;    
    p            =  [p (log(alpha_0) - sum(data_vec) *log(V))];
%                     + gammaln(V*gam) - V*gammaln(gam)...
%                     + sum(gammaln(data_vec + gam)) - gammaln(sum(data_vec) + V*gam))];     
    p            =  1 ./ sum(exp(repmat(p, [K+1 1]) - repmat(p', [1 K+1])), 2);
    k_selected   =  find(mnrnd(1, p) == 1);
end


function Likehd = Likehd_dv(v, a_d, A_d, V, K, Oa, OaC, KC, ...
    eta, theta, alpha_a, alpha_0, gam)

    %Likehd   = 0;
    v_vec    = zeros(1, V);
    v_vec(v) = 1;

    p_list   = zeros(1, sum(Oa(a_d)) + A_d);
    num      = 1;
    
    for aidx = 1 : A_d
        
        a    = a_d(aidx);
        O_a  = sum(OaC(a, :));
        
        for o = 1 : Oa(a) 
            p_list(num) = log(OaC(a, o)+eps) - log(O_a+alpha_a) - log(A_d) + log(theta(eta(a,o), v) + eps);
            num         = num + 1;
           %Likehd = Likehd + OaC(a, o)/(O_a+alpha_a) * theta(eta(a, o), v);           
        end
        
        p_list(num) = log(alpha_a) - log(O_a+alpha_a) - log(A_d) + log(Likehd_k(v_vec, V, K, KC, theta, alpha_0, gam) + eps);
        num         = num + 1;
        %Likehd = Likehd + alpha_a/(O_a+alpha_a) * Likehd_k(v_vec, V, K, KC, theta, alpha_0, gam);        
    end
    
    Likehd = sum(exp(p_list));
end


function Likehd = Likehd_k(data_vec, V, K, KC, theta, alpha_0, gam)
    Oall         = sum(KC);
    p            = log(KC+eps) - log(Oall+alpha_0+eps) ...
                        + transpose(sum(repmat(data_vec, [K 1]) .* log(theta + eps), 2));  
    pK1          = log(alpha_0+eps) - log(Oall+alpha_0+eps)...
                          - sum(data_vec) * log(V);
%                         + gammaln(V*gam) - V*gammaln(gam)...
%                         + sum(gammaln(data_vec + gam)) - gammaln(sum(data_vec) + V*gam);    
    Likehd       = sum(exp(p)) + exp(pK1) ;
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
