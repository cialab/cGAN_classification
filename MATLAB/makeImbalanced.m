% This script is highly specific

rd='./datasets';
sigmas=["sqrt2","sqrt5"];
folds=4;
classes=["non-tumor","tumor"];
ratio=0.1;

for s=1:length(sigmas)
    for f=1:folds
        nclass=zeros(length(classes),1);
        for c=1:length(classes)
            d=dir(fullfile(rd,sigmas(s),strcat('fold_',num2str(f)),classes(c),'training','*.h5'));
            n=strsplit(d.name,'.');
            nclass(c)=str2num(n{1});
        end
        
        % Compute number of tumor based on specified ratio
        ntumor=floor(nclass(find(classes=="non-tumor"))*ratio);
        dd=fullfile(rd,sigmas(s),strcat('fold_',num2str(f)),'tumor_imbalanced');
        if ~exist(dd,'dir')
            mkdir(dd)
        end
        
        % Make new pseudo-class
        d=dir(fullfile(rd,sigmas(s),strcat('fold_',num2str(f)),'tumor','training','*.h5'));
        patches=h5read(fullfile(d(1).folder,d(1).name),'/patches');
        patches=patches(:,:,:,randsample(size(patches,4),ntumor));
        if ~exist(fullfile(dd,'training'),'dir')
            mkdir(fullfile(dd,'training'));
        end
        h5create(fullfile(dd,'training',strcat(num2str(ntumor),'.h5')),'/patches',size(patches),'Datatype','uint8');
        h5write(fullfile(dd,'training',strcat(num2str(ntumor),'.h5')),'/patches',patches);

        % Copy validation set
        source=fullfile(rd,sigmas(s),strcat('fold_',num2str(f)),'tumor','validation');
        destination=fullfile(dd,'validation');
        copyfile(source,destination);
    end
end
