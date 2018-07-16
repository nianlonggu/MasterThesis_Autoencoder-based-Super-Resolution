
  
addpath(genpath('./'));

dataDir = 'resources/train_data/291';
saveDir = '../data/train/291/SHVCscale2/';
scale =2;
train_data_generate(dataDir, saveDir, scale, 'SHEVC','SHEVC');

%%
 addpath(genpath('./'));

 dataDir = 'resources/test_data/Set5';
 saveDir = '../data/test/Set5/SHVCscale2/';
 scale =2;
 test_data_generate(dataDir, saveDir, scale, 'SHEVC','SHEVC');


