function [cm] = getCM(tmap,ntmap,gt)

% Masks
pt=tmap>=ntmap;
pnt=ntmap>tmap;

tp=pt&gt;tp=sum(tp(:));
fp=pt&(~gt);fp=sum(fp(:));
fn=pnt&gt;fn=sum(fn(:));
tn=pnt&(~gt);tn=sum(tn(:));
cm=[tp fp; fn tn];

end

