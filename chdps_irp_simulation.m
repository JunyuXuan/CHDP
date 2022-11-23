%
% Cooperatively Hierachical Dirichlet Process with Superposition
%
%      using International restaurant process representation 
%
%           by simulation
%
%

function [ K_list ] = chdps_irp_simulation( A, D, V, AD, DV, alpha_0, alpha_a, alpha_d, Max_iteration )
  
    K_list  = [];    
     
    % iteration
    iter = 1;
    
    %hbb=figure('name','irp-CHDP');
    
    while iter <= Max_iteration
       
       fprintf(' --------interation num = %d \n', iter);
       
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
                       
                       % select option                    
                       [a_selected, o_selected] = Select_option(d, a_d, varphi, Oa, alpha_a);                      
                       Td(d)                    = Td(d) + 1;                       
                       varphi(d, t_selected, :) = [a_selected o_selected];
                       
                       if o_selected > Oa(a_selected)
                          
                           % select dish   
                           [k_selected]                = Select_dish(eta, K, alpha_0);  
                           Oa(a_selected)              = Oa(a_selected) + 1;
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

function [a_selected, o_selected] = Select_option(d, a_d, varphi, Oa, alpha_a)

    p_list = [];
    a_list = [];
    o_list = [];

    for aidx = 1 : length(a_d)
        
        a        = a_d(aidx);        
        idx_dt_a = find(varphi(:, :, 1) == a);        
        dtnum_a  = numel(idx_dt_a);
        
        for o = 1 : Oa(a)
            
            idx_dt_o = find(varphi(:, :, 2) == o);            
            dtnum_ao = numel(intersect(idx_dt_a, idx_dt_o));
            
            p_list = [p_list dtnum_ao/(dtnum_a+alpha_a)];
            a_list = [a_list a];
            o_list = [o_list o];
        end
        
        p_list = [p_list alpha_a/(dtnum_a+alpha_a)];
        a_list = [a_list a];
        o_list = [o_list Oa(a)+1];
    end

    p_list      =   p_list / length(a_d) ;
    idx_ao      =   find(mnrnd(1, p_list) == 1);
    
    a_selected  = a_list(idx_ao);
    o_selected  = o_list(idx_ao);
end


function [k_selected] = Select_dish(eta, K, alpha_0)

    p = [];

    if K > 0
        idx_0        = find(eta > 0); 
        tab          = tabulate(eta(idx_0));    
        p            = zeros(1, K);    
        p(tab(:, 1)) = tab(:, 2);   
    end
    
    p            =  [p alpha_0];    
    p            =  p / sum(p);
    k_selected   =  find(mnrnd(1, p) == 1);
end


