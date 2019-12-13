function extractPatches_helper(xml,slide,ps,stride,write_dir)
%EXTRACTPATCHES_HELPER Extracts patches per slide with a stride

xml_fn=strsplit(xml,'/'); xml_fn=xml_fn{end};
slide_fn=strsplit(slide,'/'); slide_fn=slide_fn{end};
fn=strsplit(xml_fn,'.');

t=0;
while ~t
    try
        openslide_load_library();
        t=1;
    catch ME
        fprintf('Unable to load openslide. Trying again.\n');
        t=0;
    end
end

t=0;
while ~t
    try
        % Get file pointer for openslide
        slidePtr = openslide_open(slide);
        t=1;
    catch ME
        fprintf('Unable to call openslide_open. Reloading library.\n');
        openslide_unload_library();
        t=0;
        tt=0;
        while ~tt
            try
                openslide_load_library();
                tt=1;
            catch ME2
                fprintf('Not able to reload. Trying again.\n');
                tt=0;
            end
        end
    end
end

fprintf('Working on %s\n',slide_fn);

% Get slide size
infoo=imfinfo(slide);
info=[infoo(1);infoo(3:length(infoo)-2)];
xSize=info(1).Width;
ySize=info(1).Height;

% Read in XML file
xDoc = xmlread(xml);

% Get annotation layers
annotations = xDoc.getElementsByTagName('Annotation');

% Go through layers
for a=0:annotations.getLength-1
    an = annotations.item(a); % dealing with one now

    % Get name of annotation
    n = an.getElementsByTagName('Attribute');
    n = n.item(0);
    name = n.getAttribute('Value');
    fprintf('Getting annotations for %s\n',char(name));

    % Get the list of regions
    regions=an.getElementsByTagName('Region');

    for regioni = 0:regions.getLength-1
        region=regions.item(regioni);  % for each region tag

        % Get a list of all the vertexes
        verticies=region.getElementsByTagName('Vertex');
        xy{regioni+1}=zeros(verticies.getLength-1,2); % allocate space for them
        for vertexi = 0:verticies.getLength-1 % iterate through all verticies
            % Get the x value of that vertex
            x=str2double(verticies.item(vertexi).getAttribute('X'));

            % Get the y value of that vertex
            y=str2double(verticies.item(vertexi).getAttribute('Y'));
            xy{regioni+1}(vertexi+1,:)=[x,y]; % finally save them into the array
        end
    end

    % No annotations found
    if isempty(xy)
        fprintf('No annotations found\nSkipping\n\n');
        continue
    end

    regions={};
    r_i=1;
    while ~isempty(xy)
        %Collect end points
        points=zeros(length(xy)*2,2);
        for i=1:length(xy)
            points(i,1)=xy{i}(1,1);
            points(i,2)=xy{i}(1,2);
            points(i+length(xy),1)=xy{i}(length(xy{i}),1);
            points(i+length(xy),2)=xy{i}(length(xy{i}),2);
        end

        % Compute distances between each end point
        p=pdist(points,'Euclidean');
        s=squareform(p);
        s=s+diag(inf(1,size(s,1)));
        minMatrix=min(s(:));
        [row,col]=find(s==minMatrix);

        % Orient the paths of the selected endpoints in the same direction
        if col(1)<=length(xy) && row(1)<=length(xy)
            xy{row(1)}=flipud(xy{row(1)});
        end
        if col(1)>length(xy) && row(1)>length(xy)
            xy{col(1)-length(xy)}=flipud(xy{col(1)-length(xy)});
        end

        % Fix indexes (see lines 66 and 67)
        if row(1)>length(xy)
            row(1)=row(1)-length(xy);
        end
        if col(1)>length(xy)
            col(1)=col(1)-length(xy);
        end

        % Merge the paths
        if row(1)==col(1)   % this means the region is complete
            regions{r_i}=xy{row(1)};
            r_i=r_i+1;
            xy={xy{1:col(1)-1},xy{col(1)+1:length(xy)}};
        else
            c=[xy{row(1)};xy{col(1)}];  % merging
            if row(1)<col(1)
                xy{row(1)}=c;
                xy={xy{1:col(1)-1},xy{col(1)+1:length(xy)}};
            else
                xy{col(1)}=c;
                xy={xy{1:row(1)-1},xy{row(1)+1:length(xy)}};
            end
        end
    end

    % Here I go patching again (new way)
    patches=cell(length(regions),1);
    for i=1:length(regions)
        region=regions{i};
        
        % Get square boundary
        min_x=min(region(:,1));
        min_y=min(region(:,2));
        max_x=max(region(:,1));
        max_y=max(region(:,2));
        
        % Read in bounding image
        im=openslide_read_region(slidePtr,int64(min_x),int64(min_y),int64(max_x-min_x),int64(max_y-min_y),'level',0);
        im=im(:,:,2:4);
                        
        % Generate query points
        cols=1:stride:max_x-min_x;
        rows=1:stride:max_y-min_y;
        [Xs,Ys] = meshgrid(cols,rows);
        set=[Xs(:) Ys(:)];

        % Recomute current region
        region(:,1)=region(:,1)-min_x;
        region(:,2)=region(:,2)-min_y;
        
        me=inpolygon(set(:,1),set(:,2),region(:,1),region(:,2));
        right=inpolygon(set(:,1)+ps-1,set(:,2),region(:,1),region(:,2));
        down=inpolygon(set(:,1),set(:,2)+ps-1,region(:,1),region(:,2));
        diago=inpolygon(set(:,1)+ps-1,set(:,2)+ps-1,region(:,1),region(:,2));
        is_in=me&right&down&diago;

        % Get legal coordinates
        X=Xs(is_in);
        Y=Ys(is_in);
        num_patches=length(X);

        if num_patches>0
            collector=zeros(ps,ps,3,num_patches,'uint8');
            for pp=1:num_patches
                col=X(pp);
                row=Y(pp);
                patch=im(row:row+ps-1,col:col+ps-1,:);
                collector(:,:,:,pp)=patch;
            end
            patches{i}=collector;
        end
    end
    patches=cat(4,patches{:});

    % Make directory to write
    if ~exist(strcat(write_dir,'/',fn{1},'/',char(name)),'dir')
        mkdir(strcat(write_dir,'/',fn{1},'/',char(name)));
    end
    
    % Write to file
    h5create(strcat(write_dir,'/',fn{1},'/',char(name),'/',num2str(size(patches,4)),'.h5'),'/patches',size(patches),'Datatype','uint8');
    h5write(strcat(write_dir,'/',fn{1},'/',char(name),'/',num2str(size(patches,4)),'.h5'),'/patches',patches);

    fprintf('Done with %s\n',slide_fn);
end

end

