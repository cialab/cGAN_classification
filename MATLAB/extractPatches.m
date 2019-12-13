rng(1);

% openslide
addpath('/isilon/datalake/cialab/scratch/cialab/tet/.usr/include/openslide');
addpath('/isilon/datalake/cialab/scratch/cialab/tet/.usr/lib');

% Configuration
ps=256;
stride=112;
write_dir='./patches';

% Get filenames
xmls=dir('./annotations/*.xml');
slides=dir('/isilon/datalake/cialab/original/cialab/image_database/d00119/marked/*.ndpi');
slides=cat(1,slides,dir('/isilon/datalake/cialab/original/cialab/image_database/d00119/unmarked/*.svs'));

% Loop around
parfor i=1:length(xmls)
    snx=strsplit(xmls(i).name,'.');
    for j=1:length(slides)
        sns=strsplit(slides(j).name,'.');
        if strcmp(snx{1},sns{1})
            % Found xml and slides
            try
                extractPatches_helper(fullfile(xmls(i).folder,xmls(i).name), ...
                    fullfile(slides(j).folder,slides(j).name),ps,stride,write_dir);
            catch ME
                fprintf('ERROR: %s\n',slides(j).name);
            end
        end
    end
end