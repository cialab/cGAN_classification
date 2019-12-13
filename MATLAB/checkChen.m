rng(1);

gts=cat(1,dir('/isilon/datalake/cialab/scratch/cialab/tet/python/p2p/tumor_budding/hpfs5+4/masks/*HE.*.png'),...
        dir('/isilon/datalake/cialab/scratch/cialab/tet/python/p2p/tumor_budding/hpfs5+4/masks/*(HE).*.png'));
gts0=dir('/isilon/datalake/cialab/scratch/cialab/tet/python/p2p/tumor_budding/hpfs5+4/masks/*_1.tif*');
gts1=dir('/isilon/datalake/cialab/scratch/cialab/tet/python/p2p/tumor_budding/hpfs5+4/masks/*_2.tif*');
chen=cat(1,dir('./chen_masks/*HE.*.png'),...
        dir('./chen_masks/*(HE).*png'));
chen0=dir('./chen_masks/*_1*.png');
chen1=dir('./chen_masks/*_2*.png');

conds={'sqrt2','sqrt5'};a='all';
sets={'testing','testing'};
sigmas=[sqrt(2) sqrt(5)];

w=2048;
h=1280;
n=50;
t0=length(gts0)+length(gts);
t1=length(gts0)+length(gts);
ps=256;

tp=0;
tn=0;
fp=0;
fn=0;
% Tumor class
for sig=1:length(sigmas)
    p=1;
    for i=1:length(gts)
        gt=imread(fullfile(gts(i).folder,gts(i).name));
        ch=imread(fullfile(chen(i).folder,chen(i).name));
        nn=1;
        while nn<n
            row=randi([1,h-ps]);
            col=randi([1,w-ps]);
            labels=gt(row:row+ps-1,col:col+ps-1);
            if sum(labels(:)==1)/(ps*ps)>0.9
                % Check Dr. Chen
                clabels=ch(row:row+ps-1,col:col+ps-1);
                clabel=(sum(clabels(:)==1)/(ps*ps))>0.5;
                if clabel==1
                    tp=tp+1;
                end
                if clabel==0
                    fn=fn+1;
                end
                p=p+1;
                nn=nn+1;
            end
        end
    end
    for i=1:length(gts1)
        gt=imread(fullfile(gts1(i).folder,gts1(i).name));
        ch=imread(fullfile(chen1(i).folder,chen1(i).name));
        nn=1;
        while nn<n
            row=randi([1,h-ps]);
            col=randi([1,w-ps]);
            labels=gt(row:row+ps-1,col:col+ps-1);

            % Check Dr. Chen
            clabels=ch(row:row+ps-1,col:col+ps-1);
            clabel=(sum(clabels(:)==1)/(ps*ps))>0.5;
            if clabel==1
                tp=tp+1;
            end
            if clabel==0
                fn=fn+1;
            end
            p=p+1;
            nn=nn+1;
        end
    end
    
end

% Non-tumor class
for sig=1:length(sigmas)
    p=1;
    for i=1:length(gts)
        gt=imread(fullfile(gts(i).folder,gts(i).name));
        ch=imread(fullfile(chen(i).folder,chen(i).name));
        nn=1;
        while nn<n
            row=randi([1,h-ps]);
            col=randi([1,w-ps]);
            labels=gt(row:row+ps-1,col:col+ps-1);
            if sum(labels(:)==0)/(ps*ps)>0.9
                % Check Dr. Chen
                clabels=ch(row:row+ps-1,col:col+ps-1);
                clabel=(sum(clabels(:)==0)/(ps*ps))>0.9;
                if clabel==1
                    tn=tn+1;
                end
                if clabel==0
                    fp=fp+1;
                end
                p=p+1;
                nn=nn+1;
            end
        end
    end
    for i=1:length(gts0)
        gt=imread(fullfile(gts0(i).folder,gts0(i).name));
        ch=imread(fullfile(chen0(i).folder,chen0(i).name));
        nn=1;
        while nn<n
            row=randi([1,h-ps]);
            col=randi([1,w-ps]);
            labels=gt(row:row+ps-1,col:col+ps-1);
            % Check Dr. Chen
            clabels=ch(row:row+ps-1,col:col+ps-1);
            clabel=(sum(clabels(:)==0)/(ps*ps))>0.9;
            if clabel==1
                tn=tn+1;
            end
            if clabel==0
                fp=fp+1;
            end
            p=p+1;
            nn=nn+1;
        end
    end

end

tp/(tp+fp)*100
tp/(tp+fn)*100
2*((tp/(tp+fp))*(tp/(tp+fn)))/((tp/(tp+fp))+(tp/(tp+fn)))*100