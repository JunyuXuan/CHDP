%
% Cooperatively Hierachical Dirichlet Process with Maximization
%
%      using International restaurant process representation
%
%           by simulation
%
%

function [ K_list ] = chdpm_irp_simulation( A, D, V, AD, DV, alpha_0, alpha_a, alpha_d, Max_iteration )
  
    K_list  = [];    
      
    % iteration
    iter = 1;
    
    %hbb=figure('name','irp-CHDP');
    
    while iter <= Max_iteration
       
       fprintf(' --------iteration num = %d \n', iter);
       
        %
        Td      = zeros(1, D);    
        phi     = zeros(D, V);
        varphi  = zeros(D, 1, 2);
        Oa      = zeros(1, A);
        eta     = zeros(A, 1);

        K   = 0;
       
        %
       for d = 1 : D
           a_d = find(AD(:, d) > 0);
           
           for n = 1 : V
               
               if DV(d,n) > 0
                   
                   % select table
                   t_selected  = Select_table(d, n, Td, phi, alpha_d);
                   phi(d, n)   = t_selected; 
                   
                   if t_selected >= Td(d) + 1
                       
                       Td(d)                        = Td(d) + 1; 
                       
                       % select option                    
                       [a_selected, o_selected]     = Select_option(d, a_d, varphi, Oa, eta, K, alpha_a, alpha_0); 
                       varphi(d, t_selected, :)     = [a_selected o_selected];
                       
                       if o_selected > Oa(a_selected)
                          
                           Oa(a_selected)              = Oa(a_selected) + 1;
                           
                           % select dish   
                           [k_selected]                = Select_dish(eta, K, alpha_0); 
                           eta(a_selected, o_selected) = k_selected;
                           
                           if k_selected > K                              
                               K = K + 1;                               
                           end                          
                           
                       end
                       
                   end
                                      
               end
                              
           end
       end
       
       K_list(iter) = K;
              
       %% iter              
       fprintf(' --------                          K  = %d        \n', K);       
       iter = iter + 1;
       
    end
    

end

function t_selected = Select_table(d, n, Td, phi, alpha_d)
    p = zeros(1, Td(d)+1);
    
    for t = 1 : Td(d)
        p(t) = length(find(phi(d, :) == t));
    end
    
    p(Td(d)+1)      =   alpha_d;
    p               =   p ./ sum(p) ;
    t_selected      =   find(mnrnd(1, p) == 1);
end

function [a_selected, o_selected] = Select_option(d, a_d, varphi, Oa, eta, K, alpha_a, alpha_0)

    Oasum  = sum(Oa(a_d));
    A_d    = length(a_d);
    dtnum  = zeros(1, A_d);

    p_list = zeros(1, Oasum+A_d);
    a_list = zeros(1, Oasum+A_d);
    o_list = zeros(1, Oasum+A_d);
    k_list = zeros(1, Oasum+A_d);
        
%     Ok     = zeros(1, K);    
%     tab    = tabulate(eta(:));
%     
%     if size(tab, 1) > 1
%         if tab(1, 1) == 0
%             Ok(tab(2:end, 1)) = tab(2:end, 2);
%         else
%             Ok(tab(:, 1)) = tab(:, 2);
%         end
%     end
%     
%     Oall    = sum(Ok);

    k_maxp  = zeros(A_d, K);
    
    pind    = 1;
        
    for ai = 1 : A_d
        
        a         = a_d(ai);        
        idx_dt_a  = find(varphi(:, :, 1) == a);        
        dtnum_a   = numel(idx_dt_a);        
        dtnum(ai) = dtnum_a;
        
        for o = 1 : Oa(a)
            
            k_ao     = eta(a, o);
            idx_dt_o = find(varphi(:, :, 2) == o);            
            dtnum_ao = numel(intersect(idx_dt_a, idx_dt_o));
            
            p_list(pind) = dtnum_ao/(dtnum_a+alpha_a);
            a_list(pind) = a;
            o_list(pind) = o;
            k_list(pind) = k_ao;
            pind         = pind + 1;
                        
            k_maxp(ai, k_ao) = k_maxp(ai, k_ao) + dtnum_ao/(dtnum_a+alpha_a);            
        end
        
        %k_maxp(ai, :) = k_maxp(ai, :) + alpha_a/(dtnum_a+alpha_a) * Ok/(Oall + alpha_0); 
        
        p_list(pind) = alpha_a/(dtnum_a+alpha_a); % alpha_a/(dtnum_a+alpha_a) * alpha_0/(Oall + alpha_0);
        a_list(pind) = a;
        o_list(pind) = Oa(a)+1;
        k_list(pind) = K+1;
        pind         = pind + 1;        
    end

%     k_s_list = zeros(A_d, K);
    
    for k = 1 : K    
        
        maxpk   = max(k_maxp(:, k));        
        ai_list = find(k_maxp(:, k) == maxpk);
        ai      = ai_list(unidrnd(length(ai_list)));
                
        if ai > 0
            a       = a_d(ai);
            idx_na  = find(a_list ~= a);
            idx_k   = find(k_list == k);
            idx_nak = intersect(idx_na, idx_k);
            
            p_list(idx_nak) = 0;
            
%             idx_a   = find(a_list == a);
%             idx_k1  = find(k_list == K+1);
%             idx_ak1 = intersect(idx_a, idx_k1);
%             
%             p_list(idx_ak1) = p_list(idx_ak1) + alpha_a/(dtnum(ai)+alpha_a) * Ok(k)/(Oall + alpha_0);
            
%             k_s_list(ai, k) = 1;
        end
    end
    
    p_list      =   p_list / sum(p_list) ;
    idx_ao      =   find(mnrnd(1, p_list) == 1);
    
    a_selected  = a_list(idx_ao);
    o_selected  = o_list(idx_ao);
%     ks_selected = k_list(idx_ao);
%     
%     if ks_selected == K + 1
%         ai          = find(a_d == a_selected);
%         ks_selected = find(k_s_list(ai, :) > 0);
%     end
    
end


function [k_selected] = Select_dish(eta, K, alpha_0)

    p  = [];
%     pp = [];
    
    if K > 0
        idx_n0       = find(eta > 0);
        tab          = tabulate(eta(idx_n0));    
        p            = zeros(1, K);    
        p(tab(:, 1)) = tab(:, 2);
%         if ~isempty(ks_selected)
%             pp              = zeros(1, K);
%             pp(ks_selected) = p(ks_selected);
%         end
    end
    
    
    p            =  [p alpha_0];    
    p            =  p / sum(p);
    k_selected   =  find(mnrnd(1, p) == 1);
end


