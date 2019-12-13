d=dir('./tb_ultimate_test annotated3/*.PNG');
[Y,ndx,dbg]=natsortfiles({d.name});
d=d(ndx);
d2=dir('X:/python/p2p/tumor_budding/hpfs5+4/*.tif');
[Y,ndx,dbg]=natsortfiles({d2.name});
d2=d2(ndx);

load('testing_hpf_labels.mat');

for i=1:270
    a=imread(fullfile(d(i).folder,d(i).name));
    if p(i)==-1
        a=a(:,65:1280-64,:); % crop
        a=(a>0);
        h=imshow(a);
        m=zeros(720,1152,'logical');
        while 1
            l=imline;
            l=createMask(l);
            if ~ishandle(h)
                break;
            end
            m=m|l;
        end
        a=l|a;
        imshow(a);
        pause;
        
    else
        imwrite(a,strcat('./chen_fixed/',d(i).name));
    end
end
        