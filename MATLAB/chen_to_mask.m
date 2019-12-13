d=dir('./tb_ultimate_test annotated3/*.PNG');
[Y,ndx,dbg]=natsortfiles({d.name});
d=d(ndx);
%d2=dir('/isilon/datalake/cialab/scratch/cialab/tet/python/p2p/tumor_budding/hpfs5+4/*.tif');
d2=dir('X:/python/p2p/tumor_budding/hpfs5+4/*.tif');
[Y,ndx,dbg]=natsortfiles({d2.name});
d2=d2(ndx);

% % Sanity check
% for i=1:length(d2)
%     im=imread(fullfile(d(i).folder,d(i).name));
%     im2=imread(fullfile(d2(i).folder,d2(i).name));
%     imshowpair(im,im2,'montage');
%     pause;
% end
% % think we're good

load('testing_hpf_labels.mat');

for i=1:270
    if p(i)==1
        a=ones(1280,2048,'uint8');
        imwrite(a,strcat('./chen_masks/',d2(i).name,'_mask.png'));
    elseif p(i)==0
        a=zeros(1280,2048,'uint8');
        imwrite(a,strcat('./chen_masks/',d2(i).name,'_mask.png'));
    else % interpret mask
        im=imread(fullfile(d(i).folder,d(i).name));
        im=im(:,65:1280-64,:);      % crop
        dots=(im(:,:,2)>171)&(im(:,:,2)<174);
        b=(im(:,:,3)>70)&~dots;
        dots=imresize(dots,[1280,2048]);
        b=imresize(b,[1280 2048]);  % resize
        b=imdilate(b,strel('disk',2,0));
        stats = regionprops(dots,'centroid');
        for j=1:length(stats)
            pp=round(stats(j).Centroid);
            b=imfill(b,[pp(2),pp(1)]);
        end
%         subplot(1,2,1);
%         imshow(b);
%         subplot(1,2,2);
%         imshow(imresize(im,[1280 2048]));
        imwrite(b,strcat('./chen_masks/',d2(i).name,'_mask.png'));
    end
end

        
        