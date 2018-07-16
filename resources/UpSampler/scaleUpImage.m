function [ upscaledPic ] = scaleUpImage(pic, factor, method)
%scaleUpImage scales up an input image with mehtod speiciefied as string in
%method and upscale factor specified in factor

switch method
    case 'HEVC'
        upscaledPic = scaleUpImageHEVC(pic,factor);
    case 'bicubic'
        upscaledPic = imresize(pic,factor,'bicubic');
    case 'bilinear'
        upscaledPic = imresize(pic,factor,'bilinear');
    case 'nearest'
        upscaledPic = imresize(pic,factor,'nearest');
    case 'SHEVC'
         % use Christians Upsampler Application for upsampling...that is
         % almost the same as the reference software does
        
        % write out picture to be downsampled
        X = zeros(size(pic)/2);
        yuvwrite(round(pic*255),'tmp'); yuvwrite(X,'tmp');yuvwrite(X,'tmp');
        
        win = num2str(size(pic,2));
        hin  = num2str(size(pic,1));
        wout = num2str(size(pic,2)*factor);
        hout = num2str(size(pic,1)*factor);
        
        doit = ['./UpSampler/UpsamplerSHVC -i ' win 'x' hin ' tmp.yuv -o ' wout 'x' hout ' UpsampledTmp.yuv' ];    
        system(doit);
        upscaledPic = yuvread('UpsampledTmp.yuv',0, size(pic,1)*factor, size(pic,2)*factor,'.','y')/255;
        system('rm tmp.yuv UpsampledTmp.yuv');

    case 'noFilter'
        upscaledPic = zeros(size(pic,1)*factor, size(pic,2)*factor);
        upscaledPic(1:factor:end, 1:factor:end )= pic;

    case 'Ideal'

        % F = fftshift(fft2(pic));
        % upscaled_spec = zeros(size(pic,1)*factor, size(pic,2)*factor);
        % cP = ceil((size(upscaled_spec)+1)/2);
        % upscaled_spec( cP(1)-floor(size(upscaled_spec,1)/(2*factor)):cP(1)+ceil(size(upscaled_spec,1)/(2*factor))-1, ...
        %       cP(2)-floor(size(upscaled_spec,2)/(2*factor)):cP(2)+ceil(size(upscaled_spec,2)/(2*factor))-1 )= F;

        % upscaledPic = abs(ifft2(ifftshift(upscaled_spec)));
        if factor ~= 2
            error( 'scaleUpImage Ideal mode only suitable for factor=2!' )
        end
        
        upscaledPic = zeros(size(pic,1)*factor, size(pic,2)*factor);

        pic_spec = fft2(pic);

        % horizontal shift
        freq = linspace(0,0.5,size(pic,2)/2+1);
        freq_leftside = -flip(freq);
        freq_overall = [freq,freq_leftside(2:end-1)];
        pic_h_shift_05 = ifft2(pic_spec.* exp(pi*1j*freq_overall),'symmetric');

        %% vertical shift
        pic_spec = fft2(pic');
        % set up a vector representing normalized frequencies
        freq = linspace(0,0.5,size(pic',2)/2+1);
        freq_leftside = -flip(freq);
        freq_overall = [freq,freq_leftside(2:end-1)];
        pic_v_shift_05 = ifft2(pic_spec.* exp(pi*1j*freq_overall),'symmetric')';

        %% horizontal and vertical shift
        pic_spec = fft2(pic_h_shift_05');
        %   set up a vector representing normalized frequencies
        freq = linspace(0,0.5,size(pic',2)/2+1);
        freq_leftside = -flip(freq);
        freq_overall = [freq,freq_leftside(2:end-1)];
        pic_hv_shift_05 = ifft2(pic_spec.* exp(pi*1j*freq_overall),'symmetric')';

        upscaledPic(1:2:end,1:2:end) = pic;
        upscaledPic(1:2:end,2:2:end) = pic_h_shift_05;
        upscaledPic(2:2:end,1:2:end) = pic_v_shift_05;
        upscaledPic(2:2:end,2:2:end) = pic_hv_shift_05;


    otherwise
        error('Invalid method specified');
end
end

