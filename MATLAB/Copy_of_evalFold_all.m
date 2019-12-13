function [results] = Copy_of_evalFold_all(c,tv)


dd='/isilon/datalake/cialab/scratch/cialab/tet/python/p2p/tumor_budding/revision_9_30/models';
conds={'sigma_sqrt2','sigma_sqrt5'};
trvl={'testing2'};
ttt='tumor';
results=cell(4,1);

d=fullfile(dd,conds{c},'all');
d_t_o=dir(fullfile(d,ttt,'test',ttt,strcat(trvl{tv},'_o*.png')));
d_t_t=dir(fullfile(d,ttt,'test',ttt,strcat(trvl{tv},'_r*.png')));
d_t_nt=dir(fullfile(d,ttt,'test','non-tumor',strcat(trvl{tv},'_r*.png')));
d_nt_t=dir(fullfile(d,'non-tumor','test',ttt,strcat(trvl{tv},'_r*.png')));
d_nt_nt=dir(fullfile(d,'non-tumor','test','non-tumor',strcat(trvl{tv},'_r*.png')));
d_nt_o=dir(fullfile(d,'non-tumor','test','non-tumor',strcat(trvl{tv},'_o*.png')));

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

end