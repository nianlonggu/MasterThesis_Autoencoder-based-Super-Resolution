function [] = test_data_generate( dataDir, saveDir , scale, downmethod, upmethod  )

mkdir(saveDir);
count = 0;
f_lst = [];
f_lst = [f_lst; dir(fullfile(dataDir, '*.jpg'))];
f_lst = [f_lst; dir(fullfile(dataDir, '*.bmp'))];

crop= scale*2;
% if strcmp( method , 'SHEVC')
%     crop = scale * 2;
% end

for f_iter = 1:numel(f_lst)
    f_info = f_lst(f_iter);
    if f_info.name == '.'
        continue;
    end
    f_path = fullfile(dataDir,f_info.name);
    disp(f_path);
    img_raw = imread(f_path);
    if size(img_raw,3)==3
        img_raw = rgb2ycbcr(img_raw);
        img_raw = img_raw(:,:,1);
    end
    
    name_split= strsplit(f_info.name,'.' );
    patch_name = sprintf('%s%s',saveDir, char(name_split(1)));
   
    img_raw = modcrop( img_raw, crop );   
    img_raw = single(img_raw)/255;
    image = img_raw;
    save(patch_name, 'image');
    
    img_lr = blur_and_downsample(img_raw, scale, downmethod);
    image= img_lr;
    save(sprintf('%s_%d_lr', patch_name, scale), 'image');
    
    img_interp = scaleUpImage( img_lr, scale, upmethod );
    image= img_interp;
    save(sprintf('%s_%d_interp', patch_name, scale), 'image');
    
    count = count + 1;
    display(count);
    
    
end