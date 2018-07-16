function [ LR ] = blur_and_downsample(im,factor, method)
% this function implements the most basic image acquisition model
% the captured image is modeled by a LSI blur and a downsampling operation
% LR = D*H*F_t * HR + noise
% with LR: low resoltion frame, HR: high resoltion frame, F_t: warping
% because of motion, H: blurring due to a PSF, D: downsampling
% 
%	parameter h not in use at the moment
%	EXAMPLE USAGE: I_LR = blur_and_downsample(I_HR,2,'SHEVC')


% im must be normalized

switch method
    case 'bicubic' 
        LR = imresize(im,1/factor, 'bicubic');
    case 'bilinear'
        LR = imresize(im,1/factor,'bilinear');
    case 'nearest'
        LR = imresize(im,1/factor,'nearest');
    case 'SHEVC'
        % use DownConverter Application shipped with SHM reference software
        % for downsampling
        
        % write out picture to be downsampled
        X = zeros(size(im)/2);
        yuvwrite(round(im*255),'tmp'); yuvwrite(X,'tmp');yuvwrite(X,'tmp');
        
        win = num2str(size(im,2));
        hin  = num2str(size(im,1));   
        wout = num2str(size(im,2)/factor);
        hout = num2str(size(im,1)/factor);
        
        doit = ['./DownSampler/TAppDownConvertStatic ' win ' ' hin ' tmp.yuv ' wout ' ' hout ' DownsampledTmp.yuv' ];                
        system(doit);
        LR = yuvread('DownsampledTmp.yuv',0, size(im,1)/factor, size(im,2)/factor,'.','y')/255;
        system('rm tmp.yuv DownsampledTmp.yuv');
    case 'Ideal'
        % F = fftshift(fft2(im));
        % cP = ceil((size(F)+1)/2);
        % L_spec = F( cP(1)-floor(size(F,1)/(2*factor)):cP(1)+ceil(size(F,1)/(2*factor))-1, ...
        %       cP(2)-floor(size(F,2)/(2*factor)):cP(2)+ceil(size(F,2)/(2*factor))-1 );
        % LR =  ifft2(ifftshift(L_spec));
        spec = fftshift(fft2(im));
        spec(size(spec,1)/factor+1+size(spec,1)/(2*factor):end,:) = 0;
        spec(:,size(spec,2)/factor+1+size(spec,2)/(2*factor):end) = 0;
        spec(1:size(spec,1)/factor+1-size(spec,1)/(2*factor),:) = 0;
        spec(:,1:size(spec,2)/factor+1-size(spec,2)/(2*factor)) = 0;
        im_filtered = ifft2(ifftshift(spec),'symmetric');
        LR = im_filtered(1:factor:end,1:factor:end);
    case 'noFilter'
        LR = im(1:factor:end, 1:factor:end);
    otherwise
        error('Invalid method specified');
        
end

end
