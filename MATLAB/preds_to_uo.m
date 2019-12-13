format shortG
for c=1:length(conds)
    for tv=1:length(trvl)
        r=results_all{c,tv};
        r1=r{1};
        r2=r{2};
        r3=r{3};
        r4=r{4};
        prs=zeros(179,1);
        ses=zeros(179,1);
        f1s=zeros(179,1);
        n=1;
        for i=1:20:3580
            tp=r1(i:i+20-1);
            fp=r3(i:i+20-1);
            fn=r2(i:i+20-1);
            tn=r4(i:i+20-1);
            prs(n)=sum(tp(:))/(sum(tp(:))+sum(fp(:)));
            ses(n)=sum(tp(:))/(sum(tp(:))+sum(fn(:)));
            f1s(n)=2*(prs(n)*ses(n))/(prs(n)+ses(n));
            n=n+1;
        end
        fprintf('%s %s %s\n',num2str(nanmean(prs(ses>0.75))*100),num2str(nanstd(prs(ses>0.75))*100),num2str(nanmedian(prs(ses>0.75))*100));
        fprintf('%s %s %s\n',num2str(nanmean(ses(ses>0.75))*100),num2str(nanstd(ses(ses>0.75))*100),num2str(nanmedian(ses(ses>0.75))*100));
        fprintf('%s %s %s\n',num2str(nanmean(f1s(ses>0.75))*100),num2str(nanstd(f1s(ses>0.75))*100),num2str(nanmedian(f1s(ses>0.75))*100));
%fprintf('%s %s %s\n',num2str(nanmean(prs_i3(ses>0.75))*100),num2str(nanstd(prs_i3(ses>0.75))*100),num2str(nanmedian(prs_i3(ses>0.75))*100));
%fprintf('%s %s %s\n',num2str(nanmean(ses_i3(ses>0.75))*100),num2str(nanstd(ses_i3(ses>0.75))*100),num2str(nanmedian(ses_i3(ses>0.75))*100));
%fprintf('%s %s %s\n',num2str(nanmean(f1s_i3(ses>0.75))*100),num2str(nanstd(f1s_i3(ses>0.75))*100),num2str(nanmedian(f1s_i3(ses>0.75))*100));
    end
end
