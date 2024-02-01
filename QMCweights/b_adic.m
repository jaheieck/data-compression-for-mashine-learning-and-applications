function [i] = b_adic(z,x,b,r)
% calculation of maximal index i between 0 and r, such that the first i 
% digits of the b-adic expansion of z and x are identical (O(r))
% Input: - z in [0,1)
%        - x in [0,1]
%        - prime number b for the b-adic expansion
%        - natural number r, which bounds the maximal index
% Output: - maximal index i, such that the first i digits of the b-adic 
%           expansion of z and x are identical (if no digit is identical,  
%           the result is 0)


% initialize output. Koeff_z and Koeff_x save the calculated coefficients
% of the expansion. z_prime and x_prime save the current portion of z and
% x, which are already taken into account
i = 0;
Koeff_z = zeros(1,r);
Koeff_x = zeros(1,r);
z_prime = 0;
x_prime = 0;

% look at coefficient by coefficent
for r_prime = 1:r
    % calculate the r_prime-th coefficient
    Koeff_z(r_prime) = round(floor((z-z_prime) * b^r_prime));
    Koeff_x(r_prime) = round(floor((x-x_prime) * b^r_prime));
    % if both coefficients are equal, raise the maximal index and update
    % the portions, which are already taken into account
    if Koeff_z(r_prime) == Koeff_x(r_prime)
        i = i + 1;
        z_prime = z_prime + Koeff_z(r_prime)/b^r_prime;
        x_prime = x_prime + Koeff_x(r_prime)/b^r_prime;
    % if the coefficients are not equal, we can stop with the current i
    else 
        break
    end
end

end