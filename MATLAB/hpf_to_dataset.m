rng(1);

load('../revision_7_29/thresholds.mat');

% Config
imgs=cat(1,dir('/isilon/datalake/cialab/scratch/cialab/tet/python/p2p/tumor_budding/hpfs5+4/*HE.tif'),...
        dir('/isilon/datalake/cialab/scratch/cialab/tet/python/p2p/tumor_budding/hpfs5+4/*(HE).tif'));
imgs0=dir('/isilon/datalake/cialab/scratch/cialab/tet/python/p2p/tumor_budding/hpfs5+4/*_1.tif');
imgs1=dir('/isilon/datalake/cialab/scratch/cialab/tet/python/p2p/tumor_budding/hpfs5+4/*_2.tif');
gts=cat(1,dir('/isilon/datalake/cialab/scratch/cialab/tet/python/p2p/tumor_budding/hpfs5+4/masks/*HE.*.png'),...
        dir('/isilon/datalake/cialab/scratch/cialab/tet/python/p2p/tumor_budding/hpfs5+4/masks/*(HE).*.png'));
gts0=dir('/isilon/datalake/cialab/scratch/cialab/tet/python/p2p/tumor_budding/hpfs5+4/masks/*_1.tif*');
gts1=dir('/isilon/datalake/cialab/scratch/cialab/tet/python/p2p/tumor_budding/hpfs5+4/masks/*_2.tif*');
%chens=dir('./chen_masks/*.png');

wd='./datasets/';

conds={'sqrt2','sqrt5'};a='all';
sets={'testing','testing'};
sigmas=[sqrt(2) sqrt(5)];

w=2048;
h=1280;
n=50;
t0=length(gts0)+length(gts);
t1=length(gts0)+length(gts);
ps=256;

% Tumor class
for sig=1:length(sigmas)
    patches=zeros(256,256,3,n*t1,'uint8');
    edges=zeros(256,256,3,n*t1,'uint8');
    p=1;
    for i=1:length(imgs)
        im=imread(fullfile(imgs(i).folder,imgs(i).name));
        er=edge(im(:,:,1),'Canny',[thresholds(1,1,1) thresholds(1,1,2)],sigmas(sig));
        eg=edge(im(:,:,2),'Canny',[thresholds(2,1,1) thresholds(2,1,2)],sigmas(sig));
        eb=edge(im(:,:,3),'Canny',[thresholds(3,1,1) thresholds(3,1,2)],sigmas(sig));
        ime=cat(3,er*255,eg*255,eb*255);

        gt=imread(fullfile(gts(i).folder,gts(i).name));
        nn=1;
        while nn<n
            row=randi([1,h-ps]);
            col=randi([1,w-ps]);
            labels=gt(row:row+ps-1,col:col+ps-1);
            if sum(labels(:)==1)/(ps*ps)>0.9
                patches(:,:,:,p)=im(row:row+ps-1,col:col+ps-1,:);
                edges(:,:,:,p)=ime(row:row+ps-1,col:col+ps-1,:);
                p=p+1;
                nn=nn+1;
            end
        end
        i
    end
    for i=1:length(imgs1)
        im=imread(fullfile(imgs1(i).folder,imgs1(i).name));
        er=edge(im(:,:,1),'Canny',[thresholds(1,1,1) thresholds(1,1,2)],sigmas(sig));
        eg=edge(im(:,:,2),'Canny',[thresholds(2,1,1) thresholds(2,1,2)],sigmas(sig));
        eb=edge(im(:,:,3),'Canny',[thresholds(3,1,1) thresholds(3,1,2)],sigmas(sig));
        ime=cat(3,er*255,eg*255,eb*255);

        gt=imread(fullfile(gts1(i).folder,gts1(i).name));
        nn=1;
        while nn<n
            row=randi([1,h-ps]);
            col=randi([1,w-ps]);
            labels=gt(row:row+ps-1,col:col+ps-1);
            patches(:,:,:,p)=im(row:row+ps-1,col:col+ps-1,:);
            edges(:,:,:,p)=ime(row:row+ps-1,col:col+ps-1,:);
            p=p+1;
            nn=nn+1;
        end
        i
    end
    p=cat(3,patches,edges);
    if ~exist(fullfile(wd,conds{sig},a,'tumor','testing2'))
        mkdir(fullfile(wd,conds{sig},a,'tumor','testing2'));
    end
    h5create(fullfile(wd,conds{sig},a,'tumor','testing2',strcat(num2str(size(p,4)),'.h5')),'/patches',size(p),'Datatype','uint8');
    h5write(fullfile(wd,conds{sig},a,'tumor','testing2',strcat(num2str(size(p,4)),'.h5')),'/patches',p);
    sig
end

% Non-tumor class
for sig=1:length(sigmas)
    patches=zeros(256,256,3,n*t0,'uint8');
    edges=zeros(256,256,3,n*t0,'uint8');
    p=1;
    for i=1:length(imgs)
        im=imread(fullfile(imgs(i).folder,imgs(i).name));
        er=edge(im(:,:,1),'Canny',[thresholds(1,2,1) thresholds(1,2,2)],sigmas(sig));
        eg=edge(im(:,:,2),'Canny',[thresholds(2,2,1) thresholds(2,2,2)],sigmas(sig));
        eb=edge(im(:,:,3),'Canny',[thresholds(3,2,1) thresholds(3,2,2)],sigmas(sig));
        ime=cat(3,er*255,eg*255,eb*255);

        gt=imread(fullfile(gts(i).folder,gts(i).name));
        nn=1;
        while nn<n
            row=randi([1,h-ps]);
            col=randi([1,w-ps]);
            labels=gt(row:row+ps-1,col:col+ps-1);
            if sum(labels(:)==0)/(ps*ps)>0.9
                patches(:,:,:,p)=im(row:row+ps-1,col:col+ps-1,:);
                edges(:,:,:,p)=ime(row:row+ps-1,col:col+ps-1,:);
                p=p+1;
                nn=nn+1;
            end
        end
        i
    end
    for i=1:length(imgs0)
        im=imread(fullfile(imgs0(i).folder,imgs0(i).name));
        er=edge(im(:,:,1),'Canny',[thresholds(1,2,1) thresholds(1,2,2)],sigmas(sig));
        eg=edge(im(:,:,2),'Canny',[thresholds(2,2,1) thresholds(2,2,2)],sigmas(sig));
        eb=edge(im(:,:,3),'Canny',[thresholds(3,2,1) thresholds(3,2,2)],sigmas(sig));
        ime=cat(3,er*255,eg*255,eb*255);

        gt=imread(fullfile(gts0(i).folder,gts0(i).name));
        nn=1;
        while nn<n
            row=randi([1,h-ps]);
            col=randi([1,w-ps]);
            labels=gt(row:row+ps-1,col:col+ps-1);
            patches(:,:,:,p)=im(row:row+ps-1,col:col+ps-1,:);
            edges(:,:,:,p)=ime(row:row+ps-1,col:col+ps-1,:);
            p=p+1;
            nn=nn+1;
        end
        i
    end
    p=cat(3,patches,edges);
    if ~exist(fullfile(wd,conds{sig},a,'non-tumor','testing2'))
        mkdir(fullfile(wd,conds{sig},a,'non-tumor','testing2'));
    end
    h5create(fullfile(wd,conds{sig},a,'non-tumor','testing2',strcat(num2str(size(p,4)),'.h5')),'/patches',size(p),'Datatype','uint8');
    h5write(fullfile(wd,conds{sig},a,'non-tumor','testing2',strcat(num2str(size(p,4)),'.h5')),'/patches',p);
    sig
end