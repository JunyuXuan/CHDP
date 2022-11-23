%
%   compute the theoritical expectation of CHDP
%
%


function [ expectation ] = chdp_expectation( A, D, AD, DV, alpha_0, alpha_a, alpha_d )

    D_A = sum(AD, 1);
    
    D_N = sum(DV, 2);

    Td  = zeros(1, D);
    
    for d = 1 : D
        
       Nd       = D_N(d);
       
       Td(d)    = sum(alpha_d ./ (alpha_d + (1:Nd) - 1));
       
    end
    
    Oa = zeros(1, A);
    
    for a = 1 : A
        
       d_idx   = find(AD(a, :) > 0);
        
       Tda     = round(sum(Td(d_idx) ./ D_A(d_idx)));
        
       Oa(a)   = sum(alpha_a ./ (alpha_a + (1:Tda) - 1));
       
    end
    
    expectation    = sum(alpha_0 ./ (alpha_0 + (1:sum(Oa)) - 1));
    
    
end


