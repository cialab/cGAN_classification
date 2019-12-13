function [results,r] = evalFold_sampling(c,s,tv)
dd='/isilon/datalake/cialab/scratch/cialab/tet/python/p2p/tumor_budding/revision_7_29/models';
sets={'fold_1','fold_2','fold_3','fold_4'};
conds={'sigma_sqrt2','sigma_sqrt5'};
trvl={'training','validation'};
results=cell(4,1);
% {tp}{fp}
% {fn}{fn}
% for c=1:length(conds)
%     for s=1:length(sets)
%         for tv=1:length(trvl)
            d=fullfile(dd,conds{c},sets{s});
            d_t_o=dir(fullfile(d,'tumor','test','tumor',strcat(trvl{tv},'_o*.png')));
            d_t_t=dir(fullfile(d,'tumor','test','tumor',strcat(trvl{tv},'_r*.png')));
            d_t_nt=dir(fullfile(d,'tumor','test','non-tumor',strcat(trvl{tv},'_r*.png')));
            d_nt_t=dir(fullfile(d,'non-tumor','test','tumor',strcat(trvl{tv},'_r*.png')));
            d_nt_nt=dir(fullfile(d,'non-tumor','test','non-tumor',strcat(trvl{tv},'_r*.png')));
            d_nt_o=dir(fullfile(d,'non-tumor','test','non-tumor',strcat(trvl{tv},'_o*.png')));

            % Sampling
            n=min(length(d_t_o),length(d_nt_o));
            n
            ti=randsample(length(d_t_o),n);
            nti=randsample(length(d_nt_o),n);
            
            % Tumor patches
            tp=zeros(n,1);
            fn=zeros(n,1);
            parfor i=1:n
                o=imread(fullfile(d_t_o(ti(i)).folder,d_t_o(ti(i)).name));
                tr=imread(fullfile(d_t_t(ti(i)).folder,d_t_t(ti(i)).name));
                ntr=imread(fullfile(d_nt_t(ti(i)).folder,d_nt_t(ti(i)).name));

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
            fp=zeros(n,1);
            tn=zeros(n,1);
            parfor i=1:n
                o=imread(fullfile(d_nt_o(nti(i)).folder,d_nt_o(nti(i)).name));
                tr=imread(fullfile(d_t_nt(nti(i)).folder,d_t_nt(nti(i)).name));
                ntr=imread(fullfile(d_nt_nt(nti(i)).folder,d_nt_nt(nti(i)).name));

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
%             fullfile(conds{c},sets{s},trvl{tv})
            r=[sum(tp(:)) sum(fp(:)) (sum(tp(:))/(sum(tp(:))+sum(fp)))
                sum(fn(:)) sum(tn(:)) (sum(tn(:))/(sum(fn(:))+sum(tn(:))))
                (sum(tp(:))/(sum(tp(:))+sum(fn(:)))) (sum(tn(:))/(sum(fp(:))+sum(tn(:)))) ...
                (sum(tp(:))+sum(tn(:)))/(sum(tp(:))+sum(tn(:))+sum(fn(:))+sum(fp(:)))];
%         end
%     end
% end
end