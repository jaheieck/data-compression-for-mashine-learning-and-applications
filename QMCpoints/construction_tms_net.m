% This is an example for the deriviation and visualization of a digital net
% based on the functions given by the "Magic Point Shop"
load 'DIGSEQ\nxmats\nx_b2_m30_s12_Cs.col'
digitalseq_b2g('init0', nx_b2_m30_s12_Cs)
P = digitalseq_b2g(12, 1024);
plot(P(5,:), P(12,:), 'r+');
axis square;
hold on
P = digitalseq_b2g(12, 1024);
plot(P(5,:), P(12,:), 'b+');

