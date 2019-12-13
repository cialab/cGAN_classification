classes=dir('./patches/*/');
classes=unique({classes.name});
classes={classes{3:end}};

numperclass=zeros(length(classes),1);
for c=1:length(classes)
    d=dir(strcat('./patches/*/',classes{c},'/*.h5'));
    nums=zeros(length(d),1);
    for i=1:length(d)
        aa=strsplit(d(i).name,'.');
        nums(i)=str2num(aa{1});
    end
    numperclass(c)=sum(nums);
    fprintf('%s: %i\n',classes{c},numperclass(c));
end