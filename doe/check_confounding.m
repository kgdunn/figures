function [R, jb, order] = check_confounding(A, B, C, D, E, F, G, int)

% Now run the set of 8 experiments and find the confounding for all effects.
% Use machine generated code from Python
% python; from itertools import combinations; list(combinations('ABCDEFG', 2)); 
% list(combinations('ABCDEFG', 3)); list(combinations('ABCDEFG', 4));
% list(combinations('ABCDEFG', 5)); list(combinations('ABCDEFG', 6));
AB=A.*B;
AC=A.*C;
AD=A.*D;
AE=A.*E;
AF=A.*F;
AG=A.*G;
BC=B.*C;
BD=B.*D;
BE=B.*E;
BF=B.*F;
BG=B.*G;
CD=C.*D;
CE=C.*E;
CF=C.*F;
CG=C.*G;
DE=D.*E;
DF=D.*F;
DG=D.*G;
EF=E.*F;
EG=E.*G;
FG=F.*G;
ABC=A.*B.*C;
ABD=A.*B.*D;
ABE=A.*B.*E;
ABF=A.*B.*F;
ABG=A.*B.*G;
ACD=A.*C.*D;
ACE=A.*C.*E;
ACF=A.*C.*F;
ACG=A.*C.*G;
ADE=A.*D.*E;
ADF=A.*D.*F;
ADG=A.*D.*G;
AEF=A.*E.*F;
AEG=A.*E.*G;
AFG=A.*F.*G;
BCD=B.*C.*D;
BCE=B.*C.*E;
BCF=B.*C.*F;
BCG=B.*C.*G;
BDE=B.*D.*E;
BDF=B.*D.*F;
BDG=B.*D.*G;
BEF=B.*E.*F;
BEG=B.*E.*G;
BFG=B.*F.*G;
CDE=C.*D.*E;
CDF=C.*D.*F;
CDG=C.*D.*G;
CEF=C.*E.*F;
CEG=C.*E.*G;
CFG=C.*F.*G;
DEF=D.*E.*F;
DEG=D.*E.*G;
DFG=D.*F.*G;
EFG=E.*F.*G;

ABCD=A.*B.*C.*D;
ABCE=A.*B.*C.*E;
ABCF=A.*B.*C.*F;
ABCG=A.*B.*C.*G;
ABDE=A.*B.*D.*E;
ABDF=A.*B.*D.*F;
ABDG=A.*B.*D.*G;
ABEF=A.*B.*E.*F;
ABEG=A.*B.*E.*G;
ABFG=A.*B.*F.*G;
ACDE=A.*C.*D.*E;
ACDF=A.*C.*D.*F;
ACDG=A.*C.*D.*G;
ACEF=A.*C.*E.*F;
ACEG=A.*C.*E.*G;
ACFG=A.*C.*F.*G;
ADEF=A.*D.*E.*F;
ADEG=A.*D.*E.*G;
ADFG=A.*D.*F.*G;
AEFG=A.*E.*F.*G;
BCDE=B.*C.*D.*E;
BCDF=B.*C.*D.*F;
BCDG=B.*C.*D.*G;
BCEF=B.*C.*E.*F;
BCEG=B.*C.*E.*G;
BCFG=B.*C.*F.*G;
BDEF=B.*D.*E.*F;
BDEG=B.*D.*E.*G;
BDFG=B.*D.*F.*G;
BEFG=B.*E.*F.*G;
CDEF=C.*D.*E.*F;
CDEG=C.*D.*E.*G;
CDFG=C.*D.*F.*G;
CEFG=C.*E.*F.*G;
DEFG=D.*E.*F.*G;

ABCDE=A.*B.*C.*D.*E;
ABCDF=A.*B.*C.*D.*F;
ABCDG=A.*B.*C.*D.*G;
ABCEF=A.*B.*C.*E.*F;
ABCEG=A.*B.*C.*E.*G;
ABCFG=A.*B.*C.*F.*G;
ABDEF=A.*B.*D.*E.*F;
ABDEG=A.*B.*D.*E.*G;
ABDFG=A.*B.*D.*F.*G;
ABEFG=A.*B.*E.*F.*G;
ACDEF=A.*C.*D.*E.*F;
ACDEG=A.*C.*D.*E.*G;
ACDFG=A.*C.*D.*F.*G;
ACEFG=A.*C.*E.*F.*G;
ADEFG=A.*D.*E.*F.*G;
BCDEF=B.*C.*D.*E.*F;
BCDEG=B.*C.*D.*E.*G;
BCDFG=B.*C.*D.*F.*G;
BCEFG=B.*C.*E.*F.*G;
BDEFG=B.*D.*E.*F.*G;
CDEFG=C.*D.*E.*F.*G;

ABCDEF=A.*B.*C.*D.*E.*F;
ABCDEG=A.*B.*C.*D.*E.*G;
ABCDFG=A.*B.*C.*D.*F.*G;
ABCEFG=A.*B.*C.*E.*F.*G;
ABDEFG=A.*B.*D.*E.*F.*G;
ACDEFG=A.*C.*D.*E.*F.*G;
BCDEFG=B.*C.*D.*E.*F.*G;

ABCDEFG = A.*B.*C.*D.*E.*F.*G;

X = [int,A,B,C,D,E,F,G,AB,AC,AD,AE,AF,AG,BC,BD,BE,BF,BG,CD,CE,CF,CG,DE,DF,DG,EF,EG,FG,...
     ABC,ABD,ABE,ABF,ABG,ACD,ACE,ACF,ACG,ADE,ADF,ADG,AEF,AEG,AFG,BCD,BCE,BCF,BCG,BDE,BDF,BDG,BEF,BEG,BFG,CDE,CDF,CDG,CEF,CEG,CFG,DEF,DEG,DFG,EFG, ...
     ABCD,ABCE,ABCF,ABCG,ABDE,ABDF,ABDG,ABEF,ABEG,ABFG,ACDE,ACDF,ACDG,ACEF,ACEG,ACFG,ADEF,ADEG,ADFG,AEFG,BCDE,BCDF,BCDG,BCEF,BCEG,BCFG,BDEF,BDEG,BDFG,BEFG,CDEF,CDEG,CDFG,CEFG,DEFG, ...
     ABCDE,ABCDF,ABCDG,ABCEF,ABCEG,ABCFG,ABDEF,ABDEG,ABDFG,ABEFG,ACDEF,ACDEG,ACDFG,ACEFG,ADEFG,BCDEF,BCDEG,BCDFG,BCEFG,BDEFG,CDEFG, ...
     ABCDEF,ABCDEG,ABCDFG,ABCEFG,ABDEFG,ACDEFG,BCDEFG, ABCDEFG];
order = cellstr({'int','A','B','C','D','E','F','G','AB','AC','AD','AE','AF','AG','BC','BD','BE','BF','BG','CD','CE','CF','CG','DE','DF','DG','EF','EG','FG',...
    'ABC','ABD','ABE','ABF','ABG','ACD','ACE','ACF','ACG','ADE','ADF','ADG','AEF','AEG','AFG','BCD','BCE','BCF','BCG','BDE','BDF','BDG','BEF','BEG','BFG','CDE','CDF','CDG','CEF','CEG','CFG','DEF','DEG','DFG','EFG', ... 
    'ABCD','ABCE','ABCF','ABCG','ABDE','ABDF','ABDG','ABEF','ABEG','ABFG','ACDE','ACDF','ACDG','ACEF','ACEG','ACFG','ADEF','ADEG','ADFG','AEFG','BCDE','BCDF','BCDG','BCEF','BCEG','BCFG','BDEF','BDEG','BDFG','BEFG','CDEF','CDEG','CDFG','CEFG','DEFG', ...
    'ABCDE','ABCDF','ABCDG','ABCEF','ABCEG','ABCFG','ABDEF','ABDEG','ABDFG','ABEFG','ACDEF','ACDEG','ACDFG','ACEFG','ADEFG','BCDEF','BCDEG','BCDFG','BCEFG','BDEFG','CDEFG', ...
    'ABCDEF','ABCDEG','ABCDFG','ABCEFG','ABDEFG','ACDEFG','BCDEFG','ABCDEFG'});
[R, jb] = rref(X);