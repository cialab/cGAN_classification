% Write and read roots
wd='./datasets';
sd='./patches';
rd='./patches_edges';

% Count slides
sd=dir(sd);
sd=sd(3:end);

% Canny stuff
load('thresholds.mat');
sigmas=[sqrt(2), sqrt(5)];
sigmass=["sqrt2","sqrt5"];

ps=256;
classes=["tumor","non-tumor"];

for s=1:length(sigmas)
    d=dir(fullfile(rd,sigmass(s)));
    d=d(3:end);

    for c=1:length(classes)
        % Count
        num=0;
        for i=1:length(d)
            fn=dir(fullfile(d(i).folder,d(i).name,classes{c},'*.h5'));
            a=strsplit(fn(1).name,'.');
            num=num+str2num(a{1});
        end

        % Collect
        patches=zeros(ps,ps,6,num,'uint8');
        num=0;
        for i=1:length(d)
            fn=dir(fullfile(d(i).folder,d(i).name,classes{c},'*.h5'));
            p=h5read(fullfile(fn(1).folder,fn(1).name),'/patches');
            a=strsplit(fn(1).name,'.');
            patches(:,:,:,num+1:num+str2num(a{1}))=p;
            num=num+str2num(a{1});
        end
        
        dd=fullfile(wd,sigmass(s),'all',classes(c),'training');
        mkdir(dd);
        h5create(fullfile(dd,strcat(num2str(num),'.h5')),'/patches',size(patches),'Datatype','uint8');
        h5write(fullfile(dd,strcat(num2str(num),'.h5')),'/patches',patches);
        
        dd=fullfile(wd,sigmass(s),'all',classes(c),'validation');
        mkdir(dd);
        h5create(fullfile(dd,strcat(num2str(num),'.h5')),'/patches',size(patches),'Datatype','uint8');
        h5write(fullfile(dd,strcat(num2str(num),'.h5')),'/patches',patches);
    end
end