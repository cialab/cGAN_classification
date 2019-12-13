% Depends on the all to already be made for dual cGANs

rd='./datasets';
wd='./datasets_incept';
sigmas=["sqrt2","sqrt5"];
classes1=["non-tumor","tumor"];
classes2=["nontumor","tumor"];
trvl1=["testing"];
trvl2=["validation"];

% Loops
for s=1:length(sigmas)
    for c=1:length(classes1)
        for tv=1:length(trvl1)
            d=dir(fullfile(rd,sigmas(s),'all',classes1(c),trvl1(tv),'*.h5'));
            patches=h5read(fullfile(d(1).folder,d(1).name),'/patches');
            patches=patches(:,:,1:3,:);
            dd=fullfile(wd,sigmas(s),'all',classes2(c),trvl2(tv));
            if ~exist(dd)
                mkdir(dd);
            end
            h5create(fullfile(dd,d(1).name),'/patches',size(patches),'Datatype','uint8');
            h5write(fullfile(dd,d(1).name),'/patches',patches);
        end
    end
end