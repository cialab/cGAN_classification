% Highly specific script
rng(1);

% Config
rd='./datasets';
sigmas=["sqrt2","sqrt5"];
ratio=0.1;

for s=1:length(sigmas)
    d=dir(fullfile(rd,sigmas(s),'all','tumor','training','*.h5'));
    n=dir(fullfile(rd,sigmas(s),'all','non-tumor','training','*.h5'));
    n=strsplit(n(1).name,'.');
    n=str2num(n{1});
    
    p=h5read(fullfile(d(1).folder,d(1).name),'/patches');
    r=randperm(size(p,4),round(n*ratio,0));
    p=p(:,:,:,r);
    
    mkdir(fullfile(rd,sigmas(s),'all','tumor_imbalanced','training'));
    mkdir(fullfile(rd,sigmas(s),'all','tumor_imbalanced','validation'));
    h5create(fullfile(rd,sigmas(s),'all','tumor_imbalanced','training',...
        strcat(num2str(size(p,4)),'.h5')),'/patches',size(p),'Datatype','uint8');
    h5create(fullfile(rd,sigmas(s),'all','tumor_imbalanced','validation',...
        strcat(num2str(size(p,4)),'.h5')),'/patches',size(p),'Datatype','uint8');
    h5write(fullfile(rd,sigmas(s),'all','tumor_imbalanced','training',...
        strcat(num2str(size(p,4)),'.h5')),'/patches',p);
    h5write(fullfile(rd,sigmas(s),'all','tumor_imbalanced','validation',...
        strcat(num2str(size(p,4)),'.h5')),'/patches',p);
end
    
    
    