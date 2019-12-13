d=dir('./patches/');
d=d(3:end);
wd='./patches_edges';

% Canny stuff
load('thresholds.mat');
sigmas=[sqrt(2), sqrt(5)];
sigmass=["sqrt2","sqrt5"];

ps=256;
classes=["tumor","non-tumor"];

parfor i=1:length(d)
    for c=1:length(classes)
        fn=dir(fullfile(d(i).folder,d(i).name,classes{c},'*.h5'));
        t=h5read(fullfile(fn(1).folder,fn(1).name),'/patches');
        patches=zeros(ps,ps,6,size(t,4),'uint8');
        patches(:,:,1:3,:)=t;
        for s=1:length(sigmas)
            p=zeros(ps,ps,3,size(patches,4),'uint8');
            for j=1:size(patches,4)
                er=edge(patches(:,:,1,j),'Canny',[thresholds(1,c,1) thresholds(1,c,2)],sigmas(s));
                eg=edge(patches(:,:,2,j),'Canny',[thresholds(2,c,1) thresholds(2,c,2)],sigmas(s));
                eb=edge(patches(:,:,3,j),'Canny',[thresholds(3,c,1) thresholds(3,c,2)],sigmas(s));
                p(:,:,:,j)=cat(3,er*255,eg*255,eb*255);
            end
            patches(:,:,4:6,:)=p;
            dd=fullfile(wd,sigmass{s},d(i).name,classes{c});
            mkdir(dd);
            h5create(fullfile(dd,fn(1).name),'/patches',size(patches),'Datatype','uint8');
            h5write(fullfile(dd,fn(1).name),'/patches',patches);
        end
    end
end