% Verify the confounding in a 7 factor, 8 experiment set

int = ones(8,1);
A1 = [-1, +1, -1, +1, -1, +1, -1, +1]';
B1 = [-1, -1, +1, +1, -1, -1, +1, +1]';
C1 = [-1, -1, -1, -1, +1, +1, +1, +1]';
D1 = A1 .* B1;
E1 = A1 .* C1;
F1 = B1 .* C1;
G1 = A1 .* B1 .* C1;

X = [int A1 B1 C1 D1 E1 F1 G1];
[R, jb] = rref(X)
jb  % a vector, 1 to 8, indicating which columns are independent

% check X'*X has 8's on the diagonals, and zeros everywhere else: true

% Now add confounding: basic check
AB = A1 .* B1;
AC = A1 .* C1;
BC = B1 .* C1;
ABC = A1 .* B1 .* C1;

X = [int A1 B1 C1 D1 E1 F1 G1 AB AC BC ABC];
[R, jb] = rref(X)
jb % a vector, 1 to 8, indicating which columns are independent
% Check the columns 9, 10, 11 and 12 have ones in the same positions as 
% columns 5, 6, 7 and 8, which indicate that they are aliases of these.

% Now do this on a larger scale: 2^7 full factors, so X is a 128x128 matrix for a full set
% But if we do 8 experiments, each factor is confounded with 15 others (16-fold confounding)
[R_1, jb1, order] = check_confounding(A1, B1, C1, D1, E1, F1, G1, int);
order(jb1)

% OK, now consider foldover. The next 8 experiments are generated with C = -C;
A2 = A1;
B2 = B1;
C2 = -C1;
D2 = D1; % wrong: A2 .* B2;
E2 = E1; % wrong: A2 .* C2;
F2 = F1; % wrong: B2 .* C2;
G2 = G1; % wrong: A2 .* B2 .* C2;
[R_2, jb2] = check_confounding(A2, B2, C2, D2, E2, F2, G2, int);
order(jb2)   % still 8 hindependent factors

% Consider adding the two sets of experiments together

A = [A1; A2];
B = [B1; B2];
C = [C1; C2];
D = [D1; D2];
E = [E1; E2];
F = [F1; F2];
G = [G1; G2];
int = [int; int];
[R_all, jb_all, order] = check_confounding(A, B, C, D, E, F, G, int);
order(jb_all)  
[R_all_unique, i, j] = unique(flipud(R_all'), 'rows');
flipped_order  = flipud(order');
flipped_order(i)
% 
% flipped_order(i) = 
%     'ACE'
%     'CG'
%     'CF'
%     'CE'
%     'CD'
%     'BC'
%     'AE'
%     'AC'
%     'G'
%     'F'
%     'E'
%     'D'
%     'C'
%     'B'
%     'A'
%     'int'

% indicates that AC, BC, CD, CE, CF and CG are now the main parameters estiamates
% the fact that sum(R_all') is a vector of 8's, indicates that there are 8 elements in the 
% confounding patterns



% Now check what happens when we switch signs

