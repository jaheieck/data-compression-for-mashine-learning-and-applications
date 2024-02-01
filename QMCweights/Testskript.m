% we test the functions in this folder for specific scenarios

% First, some examples we can calculate by hand. The expected value is
% noted afterwards. If useful, we first look at one-dimensional examples
% first.

% b_adic.m:
[i1] = b_adic(0.75,0.25,2,0); % expected value: 0 (via definition)  
[i2] = b_adic(0.75,0.25,2,1); % expected value: 0  
[i4] = b_adic(0.3,0.25,2,1); % expected value: 1
[i5] = b_adic(0.3,0.25,2,2); % expected value: 2
[i6] = b_adic(0.3,0.25,2,3); % expected value: 3
[i7] = b_adic(0.3,0.25,2,100); % expected value: 4
[i8] = b_adic(0.25,0.25,3,100); % expected value: 100 
[i9] = b_adic(1/3,0.34,3,100); % expected value: 4
[i10] = b_adic(1/11,1/11+0.01,11,100); % expected value: 1

% algorithm_2.m and algorithm_2_efficient.m:
[N_r_i1] = algorithm_2_efficient([1 2 2],3); % expected value: 5 ((111),(120),(102),(021),(012))
[N_r_i2] = algorithm_2_efficient([8 8 8 8 8],8); % expected value: 495 (12 over 4)
[N_r_i3] = algorithm_2_efficient([8 8 8 8 8],5); % expected value: 126 (9 over 4)
[N_r_i4] = algorithm_2_efficient([0 1 0 1],3); % expected value: 0 (|i|<r has no solution)
[N_r_i5] = algorithm_2_efficient([0 1 1 1],3); % expected value: 1 ((0111))
[N_r_i6] = algorithm_2_efficient([1 1 1 1],3); % expected value: 4 ((1110),(1101),(1011),(0111))
% we observe the same values for the efficient function
[N_r_i7] = algorithm_2([1 2 2],3); % expected value: 5 
[N_r_i8] = algorithm_2([8 8 8 8 8],8); % expected value: 495 
[N_r_i9] = algorithm_2([8 8 8 8 8],5); % expected value: 126 
[N_r_i10] = algorithm_2([0 1 0 1],3); % expected value: 0
[N_r_i11] = algorithm_2([0 1 1 1],3); % expected value: 1 
[N_r_i12] = algorithm_2([1 1 1 1],3); % expected value: 4 
 
% algorithm_1.m:
X = [0.1 0.2 0.8 0.5 0.9 0.2];
Y = [2 0 0 1 1 2];
% we do not split [0,1), so we calculate the same value for every z
r = 0;
[S_r1] = algorithm_1_3(0.5,X,ones(1,6),2,r); % expected value: 6
[T_r1] = algorithm_1_3(0,X,Y,2,r); % expected value: 6
[T_r2] = algorithm_1_3(0.5,X,Y,2,r); % expected value: 6
[T_r3] = algorithm_1_3(0.99,X,Y,2,r); % expected value: 6
% we split [0,1) into [0,0.5) and [0.5,1). Therefore, for every z in those
% intervals, we obtain the same values
r = 1;
[S_r4] = algorithm_1_3(0,X,ones(1,6),2,r); % expected value: 3
[S_r5] = algorithm_1_3(0.5,X,ones(1,6),2,r); % expected value: 3
[S_r6] = algorithm_1_3(0.99,X,ones(1,6),2,r); % expected value: 3
[T_r4] = algorithm_1_3(0,X,Y,2,r); % expected value: 4
[T_r5] = algorithm_1_3(0.5,X,Y,2,r); % expected value: 2
[T_r6] = algorithm_1_3(0.99,X,Y,2,r); % expected value: 2
% we split [0,1) into [0,0.25), [0.25,0.5), [0.5,0.75) and [0.75,1).
% Therefore, for every z in those intervals, we obtain the same value
r = 2;
[S_r7] = algorithm_1_3(0.1,X,ones(1,6),2,r); % expected value: 3
[S_r8] = algorithm_1_3(0.25,X,ones(1,6),2,r); % expected value: 0
[S_r9] = algorithm_1_3(0.7,X,ones(1,6),2,r); % expected value: 1
[S_r10] = algorithm_1_3(0.9,X,ones(1,6),2,r); % expected value: 2
[T_r7] = algorithm_1_3(0.1,X,Y,2,r); % expected value: 4
[T_r8] = algorithm_1_3(0.25,X,Y,2,r); % expected value: 0
[T_r9] = algorithm_1_3(0.7,X,Y,2,r); % expected value: 1
[T_r10] = algorithm_1_3(0.9,X,Y,2,r); % expected value: 1

X = [0.1 0.2 0.8 0.5 0.9 0.2; 0.3 0.2 0.5 0.9 0.4 0.8];
Y = [2 0 0 1 1 3];
% for r=1, we split one dimension [0,1) into [0,0.5) and [0.5,1). Therefore, 
% for every z in those intervals, we obtain the same values
r = 1;
[S_r11] = algorithm_1_3([0.25;0.25],X,ones(1,6),2,r); % expected value: 6
[S_r12] = algorithm_1_3([0.3;0.4],X,ones(1,6),2,r); % expected value: 6
[S_r13] = algorithm_1_3([0.75;0.25],X,ones(1,6),2,r); % expected value: 6
[S_r14] = algorithm_1_3([0.25;0.75],X,ones(1,6),2,r); % expected value: 6
[S_r15] = algorithm_1_3([0.75;0.75],X,ones(1,6),2,r); % expected value: 6
[T_r11] = algorithm_1_3([0.25;0.25],X,Y,2,r); % expected value: 8
[T_r12] = algorithm_1_3([0.3;0.4],X,Y,2,r); % expected value: 8
[T_r13] = algorithm_1_3([0.75;0.25],X,Y,2,r); % expected value: 5
[T_r14] = algorithm_1_3([0.25;0.75],X,Y,2,r); % expected value: 9
[T_r15] = algorithm_1_3([0.75;0.75],X,Y,2,r); % expected value: 6

% for r=2, we split one dimension [0,1) into [0,0.25), [0.25,0.5), [0.5,0.75) 
% and [0.75,1) or both in [0,0.5) and [0.5,1). For z in the same 
% "quarter-interval" in every dimension, we obtain the same value
r = 2;
[S_r16] = algorithm_1_3([0.25;0.25],X,ones(1,6),2,r); % expected value: 4
[S_r17] = algorithm_1_3([0.3;0.4],X,ones(1,6),2,r); % expected value: 4
[S_r18] = algorithm_1_3([0.75;0.25],X,ones(1,6),2,r); % expected value: 5
[S_r19] = algorithm_1_3([0.25;0.75],X,ones(1,6),2,r); % expected value: 3
[S_r20] = algorithm_1_3([0.75;0.75],X,ones(1,6),2,r); % expected value: 6
[T_r16] = algorithm_1_3([0.25;0.25],X,Y,2,r); % expected value: 5
[T_r17] = algorithm_1_3([0.3;0.4],X,Y,2,r); % expected value: 5
[T_r18] = algorithm_1_3([0.75;0.25],X,Y,2,r); % expected value: 5
[T_r19] = algorithm_1_3([0.25;0.75],X,Y,2,r); % expected value: 7
[T_r20] = algorithm_1_3([0.75;0.75],X,Y,2,r); % expected value: 6

% weight.m:
% we set nu=0 and do not split [0,1). Hence every choice of z will lead to
% the same value
X = [0.1 0.2 0.8 0.5 0.9 0.2];
Y = [2 0 0 1 1 2];
[W1] = weight(0,X,Y,2,2,0); % expected value: 0.25
[W2] = weight(0.5,X,Y,2,2,0); % expected value: 0.25
[W3] = weight(0.99,X,Y,2,2,0); % expected value: 0.25
% For nu=1 we split [0,1) into [0,0.5) and [0.5,1). Das heißt für z aus
% diesen Intervallen erhalten wir jeweils die gleichen Werte.
[W4] = weight(0,X,Y,2,2,1); % expected value: 1/3
[W5] = weight(0.5,X,Y,2,2,1); % expected value: 1/6
[W6] = weight(0.99,X,Y,2,2,1); % expected value: 1/6
% Für nu=2 teilen wir [0,1) in [0,0.25), [0.25,0.5), [0.5,0.75) und [0.75,1) 
% auf. Das heißt für z aus diesen Intervallen erhalten wir jeweils die gleichen Werte.
[W7] = weight(0.1,X,Y,2,2,2); % expected value: 2/3
[W8] = weight(0.25,X,Y,2,2,2); % expected value: 0
[W9] = weight(0.7,X,Y,2,2,2); % expected value: 1/6
[W10] = weight(0.9,X,Y,2,2,2); % expected value: 1/6

X = [0.1 0.2 0.8 0.5 0.9 0.2; 0.3 0.2 0.5 0.9 0.4 0.8];
Y = [2 0 0 1 1 3];
% for nu=1, we split one dimension [0,1) into [0,0.5) and [0.5,1). 
% Therefore, for every z in those intervals, we obtain the same value
[W11] = weight([0.25;0.25],X,Y,2,2,1); % expected value: 3/8
[W12] = weight([0.3;0.4],X,Y,2,2,1); % expected value: 3/8
[W13] = weight([0.75;0.25],X,Y,2,2,1); % expected value: 1/8
[W14] = weight([0.25;0.75],X,Y,2,2,1); % expected value: 11/24
[W15] = weight([0.75;0.75],X,Y,2,2,1); % expected value: 5/24
% for r=2, we split one dimension [0,1) into [0,0.25), [0.25,0.5), [0.5,0.75) 
% and [0.75,1) or both in [0,0.5) and [0.5,1). For z in the same 
% "quarter-interval" in every dimension, we obtain the same value
[W16] = weight([0.25;0.25],X,Y,2,2,2); % expected value: 1/6
[W17] = weight([0.3;0.4],X,Y,2,2,2); % expected value: 1/6
[W18] = weight([0.75;0.25],X,Y,2,2,2); % expected value: 5/12
[W19] = weight([0.25;0.75],X,Y,2,2,2); % expected value: 5/12
[W20] = weight([0.75;0.75],X,Y,2,2,2); % expected value: 1/2



% Now we look at input, which is not easy to calculate by hand. We observe, 
% that the functions do also work in a more complex setting

% b_adic.m:
[i11] = b_adic(rand,rand,10,100);  

% algorithm_2_efficient.m:
[N_r_i13] = algorithm_2_efficient(ceil(5*rand(1,100)),50);
 
% algorithm_1.m:
s = 8;
N = 10^5;
X = rand(s,N);
Y = randn(1,N);
z = rand(s,1);
b = 11;
r = 8;
[S_r21] = algorithm_1_3(z,X,ones(1,N),b,r); 
[T_r21] = algorithm_1_3(z,X,Y,b,r); 

% weight.m:
m = 10;
nu = 8;
[W21] = weight(z,X,Y,b,m,nu); 
