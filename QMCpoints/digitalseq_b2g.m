function [P] = digitalseq_b2g(s, L)
% calculation of a (t,m,s)-net P based on a digital sequence in base 2
% taken from  F. Y. Kuo and D. Nuyens from the git "The Magic Point Shop"
% Input: - natural number s, determing the dimension
%        - natural number L, determing the number of points of P
% Output: - (t,m,s)-net P (sxL-matrix) for L = b^m

% the expleneations are taken from F. Y. Kuo and D. Nuyens
% Usage:
%   1. Initialize the generator for a point set with generating matrices in the
%   array Cs of dimension s-by-m (see below for the format of the generating
%   matrices array) with digitalseq_b2g('init0', Cs)
%   Valid options for intialization are:
%     init0       the digital sequence as is, the first point is the zero vector
%     init1       the first point is changed into an all ones vector
%     initskip    skip the first point
%
%   2. Generate the next n s-vectors of the sequence, returning an array of
%   dimensions s-by-n: P = digitalseq_b2g(s, n)
%
%   3. Ask the state of the current generator with digitalseq_b2g('state')
%
% Format of generating matrices array:
% For a generating matrix in dimension j we define the columns c_1, ..., c_m as
%         [ c_{1,1}  c_{1,2}   ...  c_{1,m}  ]
%         [ c_{2,1}  c_{2,2}   ...  c_{2,m}  ]
%   C_j = [   ...      ...     ...    ...    ] = [ c_1 c_2 ... c_m ]         (*)
%         [ c_{64,1} c_{64,2}  ...  c_{64,m} ]
% This is a binary matrix of dimensions 64-by-m (for standard digital nets these
% matrices are m-by-m, for higher order nets they typicall have the form
% alpha*m-by-m for some integer (interlacing) factor alpha).
% Now represent each column c_k as an integer:
%   c_k = sum_{i=1}^{64} c_{i,k} 2^{i-1},     for k = 1, ..., m.
% I.e., least significant bits are in the top rows.
% These integers are then placed in a row vector to represent C_j like in (*).
% The array of generating matrices is then just an s-by-m array with these
% integer representations as the rows.
% After initialization the radical inverse of these integer column
% representations are taken such that the coordinate x_j can be build by xoring
% these columns and then scaling.
%
% (w) 2010, Dirk Nuyens, Department of Computer Science, KU Leuven, Belgium
%     2015 use 64 bit integers and adjusted documentation

persistent k s_max initmode n_max cur recipd Csr maxbit

if ischar(s) && strncmp(s, 'init', 4)
    Cs = L; % when intializing we expect the generating matrices as argument 2 (which is n)
    m = size(Cs, 2);
    s_max = size(Cs, 1);
    n_max = bitshift(uint64(1), m);
    Csr = bitreverse64(uint64(Cs));
    initmode = 0;
    maxbit = 64;
    recipd = pow2(-maxbit);
    k = 0;
    cur = zeros(s_max, 1, 'uint64');
    if strcmp(s, 'init0')
        k = 0;
        initmode = 0;
    elseif strcmp(s, 'init1')
        k = 0;
        initmode = 1;
    elseif strcmp(s, 'initskip')
        initmode = -1;
        k = 1;
    else
        error('I only know about ''init0'', ''init1'' and ''initskip'', what are you talking about?');
    end
    return;
elseif ischar(s) && strcmp(s, 'state')
    P.index_of_next_point = k;
    P.previous_point_as_integer = cur';
    P.s_max = s_max;
    P.maxbit = maxbit;
    P.n_max = n_max;
    P.Csr = Csr;
    return;
end

if ((k + L) > n_max) || (s > s_max)
    error('Can only generate %d points in %d dimensions', n_max, s_max);
end

P = zeros(s, L);

if (k == 0) && (initmode == 0)
    P(:, 1) = 0; si = 2; k = k + 1;
elseif (k == 0) && (initmode == 1)
    P(:, 1) = 1; si = 2; k = k + 1;
else
    si = 1;
end

for i=si:L
    c = 1;
    while bitget(k, c) == 0
        c = c + 1;
    end
    cur = bitxor(cur, Csr(1:s_max, c));
    P(:, i) = double(cur(1:s)) * recipd;
    k = k + 1;
end
