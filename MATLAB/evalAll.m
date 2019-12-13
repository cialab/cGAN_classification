dd='/isilon/datalake/cialab/scratch/cialab/tet/python/p2p/tumor_budding/canny_byChannel/models';
sets={'all'};
conds={'sigma_sqrt2','sigma_sqrt5'};
trvl={'testing'};
tv=1;
s=1;

c=1;

results=cell(4,1);

d=fullfile(dd,conds{c},sets{s});
d_t_o=dir(fullfile(d,'tumor','test','tumor',strcat(trvl{tv},'_o*.png')));
d_t_t=dir(fullfile(d,'tumor','test','tumor',strcat(trvl{tv},'_r*.png')));
d_t_nt=dir(fullfile(d,'tumor','test','nontumor',strcat(trvl{tv},'_r*.png')));
d_nt_t=dir(fullfile(d,'nontumor','test','tumor',strcat(trvl{tv},'_r*.png')));
d_nt_nt=dir(fullfile(d,'nontumor','test','nontumor',strcat(trvl{tv},'_r*.png')));
d_nt_o=dir(fullfile(d,'nontumor','test','nontumor',strcat(trvl{tv},'_o*.png')));

% Tumor patches
tp=zeros(length(d_t_o),1);
fn=zeros(length(d_t_o),1);
parfor i=1:length(d_t_o)
    o=imread(fullfile(d_t_o(i).folder,d_t_o(i).name));
    tr=imread(fullfile(d_t_t(i).folder,d_t_t(i).name));
    ntr=imread(fullfile(d_nt_t(i).folder,d_nt_t(i).name));

    dt=immse(o,tr);
    dnt=immse(o,ntr);

    if dt<dnt
        tp(i)=1;
    end
    if dnt<dt
        fn(i)=1;
    end
end

% Non-tumor patches
fp=zeros(length(d_nt_o),1);
tn=zeros(length(d_nt_o),1);
parfor i=1:length(d_nt_o)
    o=imread(fullfile(d_nt_o(i).folder,d_nt_o(i).name));
    tr=imread(fullfile(d_t_nt(i).folder,d_t_nt(i).name));
    ntr=imread(fullfile(d_nt_nt(i).folder,d_nt_nt(i).name));

    dt=immse(o,tr);
    dnt=immse(o,ntr);

    if dt<dnt
        fp(i)=1;
    end
    if dnt<dt
        tn(i)=1;
    end
end
results{1}=tp;
results{2}=fn;
results{3}=fp;
results{4}=tn;

% Print haha
fullfile(conds{c},sets{s},trvl{tv})
[sum(tp(:)) sum(fp(:))
 sum(fn(:)) sum(tn(:))]
