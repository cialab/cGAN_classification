rng(1);

% Config
d='./hpfs/testing/';
wd='./hpfs/testing/pathologist';
n=1;
ps=256;
w=2048;
h=1280;

% Get images
imgs=dir(fullfile(d,'*.tif'));
patches=zeros(ps,ps,3,length(imgs)*n,'uint8');
pp=1;
for i=1:length(imgs)
    im=imread(fullfile(imgs(i).folder,imgs(i).name));
    for j=1:n
        ww=randi([1,w-ps]);
        hh=randi([1,h-ps]);
        p=im(hh:hh+ps-1,ww:ww+ps-1,:);
        patches(:,:,:,pp)=p;
        pp=pp+1;
    end
end

r=randperm(length(imgs)*n);
patches=patches(:,:,:,r);
for i=1:size(patches,4)
    imwrite(patches(:,:,:,i),fullfile(wd,strcat(num2str(i,'%04.f'),'.png')));
end
save(fullfile(wd,'key.mat'),'r');