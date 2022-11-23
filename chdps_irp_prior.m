% 
% CHDP with superposition 
%    
%    using International restaurant process
% 
%       by posterior inference without data
%
%          note: without considering data likelihood 
%


function [ K_list ] = chdps_irp_prior( A, D, V, AD, DV, alpha_0, alpha_a, alpha_d, Max_iteration )
  
    K_list  = [];
        
    % initialization
    [Nd, Ad, Td, Oa, K, phi, lambda, varphi, eta, TdC, OaC, KC] = initialization(A, D, V, AD, DV);
    
    % iteration
    iter = 1;
    
    %hbb=figure('name','irp-CHDP');
    
    while iter <= Max_iteration
        
       fprintf(' --------interation num = %d \n', iter);
        
       % update phi
       fprintf('             update phi  \n');
       tic;
       [Td, Oa, K, phi, lambda, varphi, eta, TdC, OaC, KC] = Update_phi(V, D, A, AD, DV, Ad, ...
          Td, Oa, K, phi, lambda, varphi, eta, TdC, OaC, KC, alpha_0, alpha_a, alpha_d, iter);       
       fprintf('                                  use time : %d  \n', toc);

       % update varphi
       fprintf('             update varphi  \n');
       tic;
       [Oa, K, lambda, varphi, eta, OaC, KC] = Update_varphi2(D, A, AD, Ad, ...
          Td, Oa, K, lambda, varphi, eta, OaC, KC, alpha_0, alpha_a, iter);       
       fprintf('                                  use time : %d  \n', toc);

       % update eta 
       fprintf('             update eta  \n');
       tic;
       [K, eta, KC] = Update_eta(A, Oa, K, eta, KC, alpha_0, iter);
       fprintf('                                  use time : %d  \n', toc);

       %% output
       fprintf(' --------       iteration = %d                          K  = %d  Oall = %d Omax = %d Tall = %d Tmax = %d  \n', iter, K, sum(Oa), max(Oa), sum(Td), max(Td));
       K_list(iter) = K;
       iter         = iter + 1;
       
       %% plot       
%        plot(K_list);
%        title('K_list')        
%        drawnow;
%        hold on;
              
    end
    
end


function [Nd, Ad, Td, Oa, K, phi, lambda, varphi, eta, TdC, OaC, KC] = initialization(A, D, V, AD, DV)

    % Max number of iteration
%     Max_iteration = 5000;
        
    % number of authors of douments
    Ad =  sum(AD ~= 0, 1);
    
    % number of words of douments
    Nd =  sum(DV ~= 0, 2);
               
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% latent variables
    
    % number of dishs
    K        = 1;
    
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
    phi      = zeros(D, V);
    
    for d = 1 : D
        Tdempty = Td(d);
        for v = 1 : V
            if DV(d, v) > 0
                
                if Tdempty > 0
                    phi(d,v) = Td(d) - Tdempty + 1;
                    Tdempty = Tdempty - 1;
                else                
                    phi(d,v) = find(mnrnd(1, 1/Td(d) * ones(1, Td(d))) == 1);
                end
            end
        end
    end
        
    % TdC(d, t) = c, i.e., count of customers on table t of documents
    TdC = zeros(D, max(Td));
    
    for d = 1 : D
       for t = 1 : Td(d)
          TdC(d,t) = length(find(phi(d, :) == t));
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


function [Td, Oa, K, phi, lambda, varphi, eta, TdC, OaC, KC] = Update_phi(V, D, A, AD, DV, Ad, ...
          Td, Oa, K, phi, lambda, varphi, eta, TdC, OaC, KC, alpha_0, alpha_a, alpha_d, iter)

    %%                 
    for d = 1 : D
        
        A_d = Ad(d); 
        a_d = find(AD(:,d) > 0);        
        
        for v = 1 : V
            
            if DV(d, v) > 0
                                
                %% remove this customer
                t_old         = phi(d, v);
                TdC(d, t_old) = TdC(d, t_old) - 1;
                   
                %% select a table
                t_selected         = select_table(d, t_old, Td(d), TdC, phi, alpha_d);
                phi(d, v)          = t_selected;
                
                % sit on a new table
                if t_selected > Td(d)
                    
                    Td(d)                    = Td(d) + 1;
                    TdC(d, t_selected)       = 1;
                    
                    % select option
                    [a_selected, o_selected] = select_option([0 0], a_d, A_d, AD, lambda, varphi, OaC, Oa, alpha_a);
                    lambda(d, t_selected)    = a_selected;
                    varphi(d, t_selected)    = o_selected;
                    
                    % choose a new option
                    if o_selected > Oa(a_selected)
                        
                        Oa(a_selected)              = Oa(a_selected) + 1;
                        OaC(a_selected, o_selected) = 1;
                        
                        % select dish
                        [k_selected]                = select_dish(0, eta, KC, K, alpha_0);
                        eta(a_selected, o_selected) = k_selected;
                        
                        % order a new dish
                        if k_selected > K
                            K             = K + 1;
                            KC(K)         = 1;
                        else
                            KC(k_selected) = KC(k_selected) + 1;
                        end
                    else
                        OaC(a_selected, o_selected) = OaC(a_selected, o_selected) + 1;
                    end
                else
                    TdC(d, t_selected) = TdC(d, t_selected) + 1;
                end
                
            end
        end
        
        %% remove empty table if exist        
        idx_dt_empty_first = find(TdC(d, 1 : Td(d)) == 0, 1);
        
        if ~isempty(idx_dt_empty_first)
            
            num = idx_dt_empty_first;
            
            for t = num : Td(d)
                
                if TdC(d, t) > 0
                    idx             = find(phi(d, :) == t);                
                    phi(d, idx)     = num;
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
            idx = find(eta == k);

            if ~isempty(idx)
                eta(idx)      = num;
                KC(num)       = KC(k);
                num           = num + 1;
            end
        end

        % remove empty columns of K
        K                 = num - 1;
        KC(K+1:end)       = [];
    end    
    
end


function [Oa, K, lambda, varphi, eta, OaC, KC] = Update_varphi2(D, A, AD, Ad, ...
          Td, Oa, K, lambda, varphi, eta, OaC, KC, alpha_0, alpha_a, iter)
       
    for d = 1 : D
       
        a_d = find(AD(:, d) > 0);
        A_d = length(a_d);
        
        for t = 1 : Td(d)
           
            %% remove this table
            ao_old                     = zeros(1, 2);
            ao_old(1)                  = lambda(d, t);
            ao_old(2)                  = varphi(d, t);            
            OaC(ao_old(1), ao_old(2))  = OaC(ao_old(1), ao_old(2)) - 1;
            
            %% sample an option             
            [a_selected, o_selected] = select_option(ao_old, a_d, A_d, AD, lambda, varphi, OaC, Oa, alpha_a);
            lambda(d, t)             = a_selected;
            varphi(d, t)             = o_selected;
            
            % if choose a new option
            if o_selected > Oa(a_selected)
                
                Oa(a_selected)              = Oa(a_selected) + 1;
                OaC(a_selected, o_selected) = 1;
                
                % select dish
                [k_selected]                = select_dish(0, eta, KC, K, alpha_0);                
                eta(a_selected, o_selected) = k_selected;
                
                % if order a new dish
                if k_selected > K
                    K             = K + 1;
                    KC(K)         = 1;
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
            idx = find(eta == k);

            if ~isempty(idx)
                eta(idx)      = num;
                KC(num)       = KC(k);
                num           = num + 1;
            end
        end

        % remove empty columns of K
        K                 = num - 1;
        KC(K+1:end)       = [];
    end  
   
end


function [K, eta, KC] = Update_eta(A, Oa, K, eta, KC, alpha_0, iter)
    
    for a = 1 : A

        O_a   = Oa(a);
        
        for o = 1 : O_a
                        
            %% remove this option
            k_old      = eta(a, o);            
            KC(k_old)  = KC(k_old) - 1;

            %% select dish            
            [k_selected]    = select_dish(k_old, eta, KC, K, alpha_0);
            eta(a, o)       = k_selected;
            
            % if order a new dish
            if k_selected > K
                K             = K + 1;
                KC(K)         = 1;
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
            idx = find(eta == k);

            if ~isempty(idx)
                eta(idx)      = num;
                KC(num)       = KC(k);
                num           = num + 1;
            end
        end

        % remove empty columns of K
        K                 = num - 1;
        KC(K+1:end)       = [];
    end  
    
end


function t_selected = select_table(d, t_old, T_d, TdC, phi, alpha_d)

    p           =  log(TdC(d, 1:T_d) +eps); 
    p           =  [p log(alpha_d+eps)];
    p           =  1 ./ sum(exp(repmat(p, [T_d+1 1]) - repmat(p', [1 T_d + 1])), 2);
    t_selected  =  find(mnrnd(1, p) == 1);
end


function [a_selected, o_selected] = select_option(ao_old, a_d, A_d, AD, lambda, varphi, OaC, Oa, alpha_a)

    Oall   = sum(Oa(a_d));
    p_list = zeros(1, Oall + A_d);
    a_list = zeros(1, Oall + A_d);
    o_list = zeros(1, Oall + A_d);
    
    num = 1;

    for aidx = 1 : A_d
        
        a    = a_d(aidx);
        O_a  = sum(OaC(a, :));
                
        for o = 1 : Oa(a)    

            p_list(num) = log(OaC(a, o)+eps) - log(O_a+alpha_a + eps);            
            a_list(num) = a;
            o_list(num) = o;
            num         = num + 1;
        end
        
        p_list(num) = log(alpha_a) - log(O_a+alpha_a+eps);
        a_list(num) = a;
        o_list(num) = Oa(a)+1;
        num         = num + 1;
    end

    p           =   1 ./ sum(exp(repmat(p_list, [(num-1) 1]) - repmat(p_list', [1 (num-1)])), 2);
    idx_ao      =   find(mnrnd(1, p) == 1);
    
    a_selected  =   a_list(idx_ao);
    o_selected  =   o_list(idx_ao);
end


function [k_selected] = select_dish(k_old, eta, KC, K, alpha_0)
    p            =  log(KC+eps);    
    p            =  [p log(alpha_0+eps)]; 
    p            =  1 ./ sum(exp(repmat(p, [K+1 1]) - repmat(p', [1 K+1])), 2);
    k_selected   =  find(mnrnd(1, p) == 1);
end
