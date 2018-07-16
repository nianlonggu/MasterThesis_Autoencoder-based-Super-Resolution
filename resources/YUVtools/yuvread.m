function [Img,err] = yuvread(filename,nr,n1,n2,directory,yuv,format,precision,offset,noext)

% YUVREAD This function returns one component  (Luma/Chroma) of
%     frame 'nr' in sequence 'filename.yuv'. Without 
%     specification of 'directory' the file is searched in
%     the current working directory.
%     'yuv' indicates which component to take; the default is
%     Luminance. (yuv == [y Y u U v V s S])
%     'yuv'==(s|S) stands for Shape; for shape-files the 
%     file extension has to be '.seg'.
%
%     NEW!: YUVREAD extends the filename with '.yuv' or '.seg' only if
%     'filename' contains no dot. I.e. also files with names like
%     'otto.y' can be read.
%     NEW!: YUVREAD extends the filename with '.yuv' or '.seg' only if
%     'filename' does not contain 'yuv'. for MPEG-CfP Files.
% 
%     ATTENTION: 0 <= nr <= total number of frames - 1 (!)
% 
%     function [Img,err] = yuvread(filename,nr,n1,n2,directory,yuv,format,precision,noext)
% 
%     IN:  filename  : file filename.yuv     (string)
%          nr        : frame number          (scalar)
%          n1,n2     : frame size [n1�n2]    (scalars)
%          directory : optional directory specification
%                                            (optional string)
%          yuv       : choose Y,U,V or S     (optional string)
%                      if more than one component is given (e.g. 'yuv'),
%                      then Img will be a cell array containing the requested
%                      components
%          format    : chrominance format    (optional string)
%                      '4:2:0'  4*Y, 1*U, 1*V  (default)
%                      '4:2:2'  4*Y, 2*U, 2*V
%                      <arbitrary other string> no chroma components
%          precision : data type (see fread help)  (default = uint8)
%          offset    : file offset (header size)   (default = 0)
%          noext     : do not append extension     (default = true)
% 
%     OUT: Img       : luminance frame    (2D-matrix)
%          err       : >0:ok, -1 = Error
% 
%     Example: 
%          Img = yuvread('dancer_dec',10,288,352);
%          Img = yuvread('dancer_dec',10,288,352,'~/tmp');
%          Img = yuvread('dancer_dec',10,288,352,'~/tmp','y');
%          Shp = yuvread('dancer_dec',10,288,352,'~/tmp','s');
%          Shp = yuvread('dancer_dec',10,288,352,'~/tmp','s','4:2:2');
%
%          Img = yuvread('dancer_dec.yuv',10,288,352,'.','y');
%
%          Img = yuvread('/original/yuv/BUS_176x144_15_orig_01.yuv',0,144,176,'.','yuv','4:2:0','uint8',0,true)
% 
%     See also IENTREAD

% --------------------------------------------------------
% Institut f�r Elektrische Nachrichtentechnik, RWTH Aachen
% Author: Mathias Wien
% Date  : 26.03.98, 
% rev.  : 19.06.98 (Chroma)
% rev.  : 29.06.98 (Shape )
% rev.  : 12.10.98 (Chrominance Format)
% rev.  : 26.08.99 (EOF Abfangen mit pcount)
% rev.  : 03.09.99 (Error handling)
% rev.  : 06.10.00 (File Extensions)
% rev.  : 13.06.01 (also pure Luminance Files)
% rev.  : 10.07.01 (also files without extension and with 'yuv' in the name)
% rev.  : 30.03.05 rusert  load multiple components simultaneously
% rev.  : 02.05.05 rusert  precision can be specified
% rev.  : 19.05.06 rusert  bugfix: multiple components + optional directory
% rev.  : 16.08.06 rusert  new flag: noext
% rev.  : 14.09.06 rusert  file offset can be specified
%                          noext is true by default
% --------------------------------------------------------

% check arguments
if nargin < 10,
  noext = true;
  if nargin < 9,
    offset = 0;
  end;
end;
if ~isempty(findstr(filename,'.')) | ~isempty(findstr(filename,'yuv')) | ...
      ~isempty(findstr(filename,'YUV')) | noext
  bext = logical(1);
else
  bext = logical(0);
end

if nargout == 2
  errflag = logical(1);                 % externe Fehlerbehandlung
else
  errflag = logical(0);                 % keine externe Fehlerbehandlung
end
err = 1;

if nargin == 0
  help yuvread
  Img = [];
  return
end

if nargin > 4
  actdir = cd;
  cd(directory)
  directory = cd;
end

if nargin < 6
  yuv = 'y';
end

if nargin < 7
  format = '4:2:0';
end

if nargin < 8
  precision = 'uchar';
end

switch format
  case '4:2:0'
    f = 0.25 ;
  case '4:2:2'
    f = 0.5;
  otherwise
    f = 0;
end;


% handle multiple components
if length(yuv)>1,
  for i = 1 : length(yuv);
    [Img{i}, er] = yuvread(filename,nr,n1,n2,directory,yuv(i),format,precision,offset,noext);
    if er == -1,
      err = -1;
    end;
  end;
  if nargin > 4
    cd(actdir);
  end
  if err == -1,
    if errflag
      err = -1;
      Img = [];
    else
      error('Read error!');
    end
  end
  return
end;


% get data from file
if ~(yuv=='s'|yuv=='S')
  pos = (nr*n1*n2*(1+2*f) + (yuv=='u'|yuv=='U')*n1*n2 + ...
      (yuv=='v'|yuv=='V')*n1*n2*(1+f)) * sizeof(precision);
  if ~(yuv=='y' | yuv=='Y')
    n1 = n1*(2*f);
    n2 = n2/2;
  end
else
  pos = nr*n1*n2*sizeof(precision);
end
pos = pos + offset;

if yuv=='s'|yuv=='S'
  if bext
    fid = fopen(filename,'r');    
  else
    fid = fopen([filename,'.seg'],'r');
  end
  if fid < 0
    if nargin > 4
      cd(actdir);
    end
    if errflag
      err = -1;
      Img = [];
      return
    else
      error('File not found!');
    end
  end
  seekerr = fseek(fid,pos,'bof');
  if seekerr == -1
    fclose(fid);
    if nargin > 4
      cd(actdir);
    end
    if errflag
      err = -1;
      Img = [];
      return
    else
      error('Frame number exceedes file size!')
    end
  end
  [Img,pcount] = fread(fid,n1*n2,precision);
  fclose(fid);
else
  if bext
    fid = fopen(filename,'r');    
  else
    fid = fopen([filename,'.yuv'],'r');
  end
  if fid < 0
    if nargin > 4
      cd(actdir);
    end
    if errflag
      err = -1;
      Img = [];
      return
    else
      error('File not found!');
    end
  end
  seekerr = fseek(fid,pos,'bof');
  if seekerr == -1
    fclose(fid);
    if nargin > 4
      cd(actdir);
    end
    if errflag
      err = -1;
      Img = [];
      return
    else
      error('Frame number exceedes file size!')
    end
  end
  [Img,pcount] = fread(fid,n1*n2,precision);
  fclose(fid);
end

if pcount == 0
  if nargin > 4
    cd(actdir);
  end
  if errflag
    err = -1;
    Img = [];
    return
  else
    error('Frame number exceedes file size!')
  end
end

Img = reshape(Img,n2,n1);
Img = Img';

if nargin > 4
  cd(actdir)
end

