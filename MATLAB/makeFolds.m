rng(1);

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
nfolds=4;
aa=round(linspace(1,length(sd),nfolds+1));

r=randperm(length(sd));
for s=1:length(sigmas)
    d=dir(fullfile(rd,sigmass(s)));
    d=d(3:end);
    d=d(r);
    for f=1:nfolds
        vld=d(aa(f):aa(f+1)-1);
        trd=cat(1,d(1:aa(f)-1),d(aa(f+1):end));
        trvld={trd,vld};
        trvl=["training","validation"];

        for tv=1:length(trvl)
            for c=1:length(classes)

                % Count
                num=0;
                for i=1:length(trvld{tv})
                    fn=dir(fullfile(trvld{tv}(i).folder,trvld{tv}(i).name,classes{c},'*.h5'));
                    a=strsplit(fn(1).name,'.');
                    num=num+str2num(a{1});
                end

                % Collect
                patches=zeros(ps,ps,6,num,'uint8');
                num=0;
                for i=1:length(trvld{tv})
                    fn=dir(fullfile(trvld{tv}(i).folder,trvld{tv}(i).name,classes{c},'*.h5'));
                    p=h5read(fullfile(fn(1).folder,fn(1).name),'/patches');
                    a=strsplit(fn(1).name,'.');
                    patches(:,:,:,num+1:num+str2num(a{1}))=p;
                    num=num+str2num(a{1});
                end
                dd=fullfile(wd,sigmass(s),strcat('fold_',num2str(f)),classes(c),trvl{tv});
                mkdir(dd);
                h5create(fullfile(dd,strcat(num2str(num),'.h5')),'/patches',size(patches),'Datatype','uint8');
                h5write(fullfile(dd,strcat(num2str(num),'.h5')),'/patches',patches);

            end
        end
    end
end