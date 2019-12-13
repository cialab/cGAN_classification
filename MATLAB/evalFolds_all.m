% These are already in the called eval function
dd='/isilon/datalake/cialab/scratch/cialab/tet/python/p2p/tumor_budding/revision_7_29/models';
conds={'sigma_sqrt2','sigma_sqrt5'};
trvl={'training','testing'};

% Loop around
results_all=cell(length(conds),length(trvl));
for c=1:length(conds)
    for tv=1:length(trvl)
        r=evalFold_all(c,tv);
        results_all{c,tv}=r;
        [c,tv]
    end
end