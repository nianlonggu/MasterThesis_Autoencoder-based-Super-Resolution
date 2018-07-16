function [] = train_data_generate( dataDir, saveDir , scale, downmethod, upmethod  )
mkdir(saveDir);
count = 0;
f_lst = [];
f_lst = [f_lst; dir(fullfile(dataDir, '*.jpg'))];
f_lst = [f_lst; dir(fullfile(dataDir, '*.bmp'))];

crop= scale*2;
%if strcmp( method , 'SHEVC')
%    crop = scale * 2;
%end

for f_iter = 1:numel(f_lst)
%     disp(f_iter);
    f_info = f_lst(f_iter);
    if f_info.name == '.'
        continue;
    end
    f_path = fullfile(dataDir,f_info.name);
    img_raw = imread(f_path);
    if size(img_raw,3)==3
        img_raw = rgb2ycbcr(img_raw);
        img_raw = img_raw(:,:,1);
    end
    
    img_raw = modcrop( img_raw, crop );
    
    img_raw = single(img_raw)/255;

    
    img_lr = blur_and_downsample(img_raw, scale, downmethod);
    img_interp = scaleUpImage( img_lr, scale, upmethod );

    img_raw_r90 = imrotate( img_raw, 90  );
    img_lr_r90 = blur_and_downsample(img_raw_r90, scale, downmethod);
    img_interp_r90 = scaleUpImage( img_lr_r90, scale, upmethod );

    img_raw_f_r0 = fliplr( img_raw );
    img_lr_f_r0 = blur_and_downsample( img_raw_f_r0, scale, downmethod  );
    img_interp_f_r0 = scaleUpImage( img_lr_f_r0, scale, upmethod  );

    img_raw_f_r90 = fliplr( imrotate( img_raw, 90 ) );
    img_lr_f_r90 = blur_and_downsample( img_raw_f_r90, scale, downmethod  );
    img_interp_f_r90 = scaleUpImage( img_lr_f_r90, scale, upmethod  );         

    patch_size = 60;
    stride = 60;
    
    img_size = size(img_raw);
    x_size = (img_size(2)-patch_size)/stride+1;
    y_size = (img_size(1)-patch_size)/stride+1;
    for x = 0:x_size-1
        for y = 0:y_size-1
            x_coord = x*stride; y_coord = y*stride; 
            patch_name = sprintf('%s%d',saveDir, count);
            
            patch = img_raw(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:);
            save(patch_name, 'patch');
            patch = img_lr(y_coord/scale+1:y_coord/scale+patch_size/scale,x_coord/scale+1:x_coord/scale+patch_size/scale,:);
            save(sprintf('%s_%d_lr', patch_name, scale), 'patch');
            patch = img_interp(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:);
            save(sprintf('%s_%d_interp', patch_name, scale), 'patch');
            
            count = count+1;            
            
        end
    end

    img_size = size(img_raw_r90);
    x_size = (img_size(2)-patch_size)/stride+1;
    y_size = (img_size(1)-patch_size)/stride+1;
    for x = 0:x_size-1
        for y = 0:y_size-1
            x_coord = x*stride; y_coord = y*stride; 
            patch_name = sprintf('%s%d',saveDir, count);
            
            patch = img_raw_r90(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:);
            save(patch_name, 'patch');
            patch = img_lr_r90(y_coord/scale+1:y_coord/scale+patch_size/scale,x_coord/scale+1:x_coord/scale+patch_size/scale,:);
            save(sprintf('%s_%d_lr', patch_name, scale), 'patch');
            patch = img_interp_r90(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:);
            save(sprintf('%s_%d_interp', patch_name, scale), 'patch');
            
            count = count+1;

        end
    end


    img_size = size(img_raw_f_r0);
    x_size = (img_size(2)-patch_size)/stride+1;
    y_size = (img_size(1)-patch_size)/stride+1;
    for x = 0:x_size-1
        for y = 0:y_size-1
            x_coord = x*stride; y_coord = y*stride; 
            patch_name = sprintf('%s%d',saveDir, count);
            
            patch = img_raw_f_r0(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:);
            save(patch_name, 'patch');
            patch = img_lr_f_r0(y_coord/scale+1:y_coord/scale+patch_size/scale,x_coord/scale+1:x_coord/scale+patch_size/scale,:);
            save(sprintf('%s_%d_lr', patch_name, scale), 'patch');
            patch = img_interp_f_r0(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:);
            save(sprintf('%s_%d_interp', patch_name, scale), 'patch');
            
            count = count+1;
        end
    end

    img_size = size(img_raw_f_r90);
    x_size = (img_size(2)-patch_size)/stride+1;
    y_size = (img_size(1)-patch_size)/stride+1;
    for x = 0:x_size-1
        for y = 0:y_size-1
            x_coord = x*stride; y_coord = y*stride; 
            patch_name = sprintf('%s%d',saveDir, count);
            
            patch = img_raw_f_r90(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:);
            save(patch_name, 'patch');
            patch = img_lr_f_r90(y_coord/scale+1:y_coord/scale+patch_size/scale,x_coord/scale+1:x_coord/scale+patch_size/scale,:);
            save(sprintf('%s_%d_lr', patch_name, scale), 'patch');
            patch = img_interp_f_r90(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:);
            save(sprintf('%s_%d_interp', patch_name, scale), 'patch');
            
            count = count+1;

        end
    end
    
    display(count);
    
    
end
