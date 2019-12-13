% These are already in the called eval function
dd='/isilon/datalake/cialab/scratch/cialab/tet/python/p2p/tumor_budding/revision_7_29/models';
sets={'fold_1','fold_2','fold_3','fold_4'};
conds={'sigma_sqrt2','sigma_sqrt5'};
trvl={'training','validation'};

% Loop around
rng(1);
results_sampling=cell(length(conds),length(sets),length(trvl));
for c=1:length(conds)
    for s=1:length(sets)
        for tv=1:length(trvl)
            [~,r]=evalFold_sampling(c,s,tv);
            results_sampling{c,s,tv}=r;
            [c,s,tv]
        end
    end
end
save('fold_results_sampling.mat','results_sampling');

% Loop around
results_all=cell(length(conds),length(sets),length(trvl));
for c=1:length(conds)
    for s=1:length(sets)
        for tv=1:length(trvl)
            [~,r]=evalFold(c,s,tv);
            results_all{c,s,tv}=r;
            [c,s,tv]
        end
    end
end
save('fold_results_all.mat','results_all');