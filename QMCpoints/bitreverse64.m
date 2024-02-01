function [v] = bitreverse64(k)
% reverses the digits of k. 
% look Stanford bit hacks: http://graphics.stanford.edu/~seander/bithacks.html
% taken from F. Y. Kuo and D. Nuyens from the git "The Magic Point Shop"
% Input: - natural number k
% Output: - natural number v with reversed digits of k

v = uint64(k);
% reverse even and odd digits
% v = ((v >> 1) & 0x55555555) | ((v & 0x55555555) << 1);
a = uint64(1431655765); a = a + pow2(32)*a;
v = bitxor( bitand( bitshift(v, -1) , a ) , ...
            bitshift( bitand(v, a) , 1 ) );
% reverse consecutive pairs
% v = ((v >> 2) & 0x33333333) | ((v & 0x33333333) << 2);
a = uint64(858993459); a = a + pow2(32)*a;
v = bitxor( bitand( bitshift(v, -2), a) , ...
            bitshift( bitand(v, a), 2 ) );
% reverse "nibbles"
% v = ((v >> 4) & 0x0F0F0F0F) | ((v & 0x0F0F0F0F) << 4);
a = uint64(252645135); a = a + pow2(32)*a;
v = bitxor( bitand( bitshift(v, -4), a) , ...
            bitshift( bitand(v, a), 4 ) );
% reverse bytes
% v = ((v >> 8) & 0x00FF00FF) | ((v & 0x00FF00FF) << 8);
a = uint64(16711935); a = a + pow2(32)*a;
v = bitxor( bitand( bitshift(v, -8), a) , ...
            bitshift( bitand(v, a), 8 ) );
% reverse 2-byte long pairs
% v = ((v >> 8) & 0x00FF00FF) | ((v & 0x00FF00FF) << 8);
a = uint64(65535); a = a + pow2(32)*a;
v = bitxor( bitand( bitshift(v, -16), a) , ...
            bitshift( bitand(v, a), 16 ) );
% reverse 4-byte long pairs
% v = (v >> 32) | (v << 32);
v = bitxor( bitshift(v, -32) , ...
            bitshift(v, 32) );
