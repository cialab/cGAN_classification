% Training non-tumor
d=dir('./datasets/sqrt2/all/non-tumor/training/*.h5');
patches=h5read(fullfile(d(1).folder,d(1).name),'/patches');
patches=patches(:,:,1:3,:);
patches=patches(:,:,:,randsample(size(patches,4),3000));
mkdir('./datasets_incept/toy/non-tumor/training');
h5create('./datasets_incept/toy/non-tumor/training/3000.h5','/patches',size(patches),'Datatype','uint8');
h5write('./datasets_incept/toy/non-tumor/training/3000.h5','/patches',patches);

% Training tumor
d=dir('./datasets/sqrt2/all/tumor/training/*.h5');
patches=h5read(fullfile(d(1).folder,d(1).name),'/patches');
patches=patches(:,:,1:3,:);
patches=patches(:,:,:,randsample(size(patches,4),3000));
mkdir('./datasets_incept/toy/tumor/training');
h5create('./datasets_incept/toy/tumor/training/3000.h5','/patches',size(patches),'Datatype','uint8');
h5write('./datasets_incept/toy/tumor/training/3000.h5','/patches',patches);

copyfile('./datasets_incept/toy/tumor/training','./datasets_incept/toy/tumor/validation');
copyfile('./datasets_incept/toy/non-tumor/training','./datasets_incept/toy/non-tumor/validation');
copyfile('./datasets_incept/toy/tumor/training','./datasets_incept/toy/tumor/testing');
copyfile('./datasets_incept/toy/non-tumor/training','./datasets_incept/toy/non-tumor/testing');