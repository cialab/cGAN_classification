function [cm] = getCM_pure(tmap,ntmap,gtg)

% Masks
pt=tmap>=ntmap;
pnt=ntmap>tmap;

tp=pt&(gtg==1);tp=sum(tp(:));
fp=pt&(gtg==0);fp=sum(fp(:));
fn=pnt&(gtg==1);fn=sum(fn(:));
tn=pnt&(gtg==0);tn=sum(tn(:));
cm=[tp fp; fn tn];

end

