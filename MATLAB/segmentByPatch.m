function [tmaps,ntmaps,tmaps2,ntmaps2,cm1,cm2] = segmentByPatch(pd,gt)

% Config
% pd='/isilon/datalake/cialab/scratch/cialab/tet/python/p2p/tumor_budding/hpfs2/bypatch2/sqrt5';
% gt='/isilon/datalake/cialab/scratch/cialab/tet/python/p2p/tumor_budding/hpfs2/gt/TB1-HE_1.tif_mask.png';

% Slide
sl=strsplit(gt,{'/','.'});
sl=sl{end-2};

% Params
w=2048;
h=1280;
ps=256;
s=32;
cols=1:s:w-ps+s;
rows=1:s:h-ps+s;
gt=imread(gt);

% Get filenames
tds=dir(fullfile(pd,sl,'t*r*.png'));
ntds=dir(fullfile(pd,sl,'nt*r*.png'));
ods=dir(fullfile(pd,sl,'t*o*.png'));

% Loopy
tmap=zeros(h,w,'single');
ntmap=zeros(h,w,'single');
tmap2=zeros(h,w,'single');
ntmap2=zeros(h,w,'single');
cmap=zeros(h,w,'single');
i=1;
for c=1:length(cols)
    for r=1:length(rows)

        t=imread(fullfile(tds(i).folder,tds(i).name));
        nt=imread(fullfile(ntds(i).folder,ntds(i).name));
        po=imread(fullfile(ods(i).folder,ods(i).name));

        dt=immse(po,t);
        dnt=immse(po,nt);
        v=abs(dt-dnt)/(dt+dnt);
        if dt<dnt
            tmap(rows(r):rows(r)+ps-1,cols(c):cols(c)+ps-1)=...
            tmap(rows(r):rows(r)+ps-1,cols(c):cols(c)+ps-1)+1;
            tmap2(rows(r):rows(r)+ps-1,cols(c):cols(c)+ps-1)=...
            tmap2(rows(r):rows(r)+ps-1,cols(c):cols(c)+ps-1)+v;
        end
        if dnt<dt
            ntmap(rows(r):rows(r)+ps-1,cols(c):cols(c)+ps-1)=...
            ntmap(rows(r):rows(r)+ps-1,cols(c):cols(c)+ps-1)+1;
            ntmap2(rows(r):rows(r)+ps-1,cols(c):cols(c)+ps-1)=...
            ntmap2(rows(r):rows(r)+ps-1,cols(c):cols(c)+ps-1)+v;
        end
        cmap(rows(r):rows(r)+ps-1,cols(c):cols(c)+ps-1)=...
            cmap(rows(r):rows(r)+ps-1,cols(c):cols(c)+ps-1)+1;
        i=i+1;
    end
end
tmaps=tmap./cmap;
ntmaps=ntmap./cmap;
tmaps2=tmap2./cmap;
ntmaps2=ntmap2./cmap;
cm1=getCM(tmap,ntmap,gt);
cm2=getCM(tmap2,ntmap2,gt);

end

