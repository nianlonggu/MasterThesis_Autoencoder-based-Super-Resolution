function yuvwrite(img, filename, b_shape, append, ext)
% YUVWRITE - Schreibt eine oder mehrere Komponenten zeilenweise in ein
%     yuv-File. Wenn das File existiert, wird der Frame angehaengt, falls 
%     append==true (default). Da keine Header oder sonstigen Informationen
%     ueber die Framegroessen in YUV-Files existieren, gibt es keine
%     Warnungen, Pruefungen zum verwendeten File! Der Datentyp beim Schreiben 
%     ist uint8 (d.h. evtl. Rundung). Wenn b_shape gesetzt ist, wird ein
%     Shape-File erzeugt (name.seg). Diese enthalten nur eine
%     (Y-)Komponente, daher wird append auf false gesetzt, falls es nicht
%     explizit angegeben wird. extension gibt an, ob eine Dateiendung 
%     angeh�ngt werden soll.
%
% Write or append an Y,U,or V-image into a YUV-File.
%     Data type: unit8, if b_shape=~0: 'filename'.seg
%   
%     function yuvwrite(img, filename, b_shape, append, extension)
%
%     IN  : img      : Frame    [2D-matrix or cell array of 2D-matrices]
%           filename : Filename [char]
%           b_shape  :          [bool] (false)
%           append   :          [bool] (true)
%           extension:          [bool] (true)

% ------------------------------------------------------------------
% Institut f�r Elektrische Nachrichtentechnik, RWTH Aachen
% Author: Mathias Wien
% Date  : 99/11/08
% rev   : 02/07/01 Thomas Rusert   append flag, multiple components,
%                                  extension  
% ------------------------------------------------------------------
if nargin<5,
  if nargin<4,
    if nargin<3,
      b_shape = ~1;
    end;
    if b_shape,
      append = ~1;
    else
      append = 1;
    end;
  end;
  ext = 1;
end;

if ext,
  if b_shape,
    fullname = [filename, '.seg'];
  else
    fullname = [filename, '.yuv'];
  end;
else
  fullname = filename;
end;

if append,
  mode = 'ab';
else
  mode = 'wb';
end;

fid = fopen(fullname, mode);

if fid == -1
  error('write not possible!')
end

if iscell(img),
  if b_shape,
    error('only one component for shape files allowed');
  end;
  for i = 1:length(img),
    temp = img{i}';
    temp = max(0, min(255, round(temp(:))));
    fwrite(fid, temp, 'uint8');
  end;
else
  img = img';
  img = max(0, min(255, round(img(:))));

  fwrite(fid, img, 'uint8');
end;

fclose(fid);




