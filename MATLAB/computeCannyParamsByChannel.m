rd='./patches';
classes={'tumor','non-tumor'};
slides=dir(fullfile(rd,'*E'));
 
c1=cell(length(slides)*length(classes),1);
c2=cell(length(slides)*length(classes),1);
c3=cell(length(slides)*length(classes),1);
for s=1:length(slides)
     for c=1:length(classes)
        d=dir(fullfile(rd,slides(s).name,classes{c},'*.h5'));
        h=h5read(fullfile(d(1).folder,d(1).name),'/patches');
        lows1=zeros(size(h,4),1);
        highs1=zeros(size(h,4),1);
        lows2=zeros(size(h,4),1);
        highs2=zeros(size(h,4),1);
        lows3=zeros(size(h,4),1);
        highs3=zeros(size(h,4),1);
        parfor i=1:size(h,4)
            p=uint8(h(:,:,:,i));
            [lows1(i),highs1(i)]=getThresholds(p(:,:,1),'canny');
            [lows2(i),highs2(i)]=getThresholds(p(:,:,2),'canny');
            [lows3(i),highs3(i)]=getThresholds(p(:,:,3),'canny');
        end
        c1{(s-1)*length(classes)+c}=[lows1;highs1];
        c2{(s-1)*length(classes)+c}=[lows2;highs2];
        c3{(s-1)*length(classes)+c}=[lows3;highs3];
        (s-1)*length(classes)+c
    end
end

t_c1=c1(1:2:48);
t_c2=c2(1:2:48);
t_c3=c3(1:2:48);
nt_c1=c1(2:2:48);
nt_c2=c2(2:2:48);
nt_c3=c3(2:2:48);
a={t_c1,t_c2,t_c3,nt_c1,nt_c2,nt_c3};
d=zeros(6,2);
for i=1:length(a)
    b=a{i};
    lows=[];
    highs=[];
    for j=1:length(b)
        c=b{j};
        l=c(1:length(c)/2);
        h=c(length(c)/2+1:end);
        lows=cat(1,lows,l);
        highs=cat(1,highs,h);
    end
    d(i,:)=[mean(lows) mean(highs)];
end

% Channel, class, low/high
thresholds=zeros(3,2,2);  
thresholds(1,1,:)=d(1,:);
thresholds(2,1,:)=d(2,:);
thresholds(3,1,:)=d(3,:);
thresholds(1,2,:)=d(4,:);
thresholds(2,2,:)=d(5,:);
thresholds(3,2,:)=d(6,:);

save('thresholds.mat','thresholds');
% t r
% t g
% t b
% nt r
% nt g
% nt b

% rd='/fs/scratch/osu8705/tumor_budding_HPFs/';
% hpfs=dir(fullfile(rd,'*.tif'));
% th=zeros(length(hpfs),2);
% for i=1:length(hpfs)
%     im=imread(fullfile(hpfs(i).folder,hpfs(i).name));
%     [th(i,1),th(i,2)]=getThresholds(rgb2gray(im),'canny');
% end