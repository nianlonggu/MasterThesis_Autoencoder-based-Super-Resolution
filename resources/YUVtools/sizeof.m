function bytes = sizeof( type );

% SIZEOF - Get size of data type in bytes.
%     Currently only platform independent data types listed in 'fread'
%     description are supported.
%
%     function bytes = sizeof( type );

% revisions
% 2005/05/02 rusert   initial version

type = lower(type);

if     strcmp(type,'schar')   || strcmp(type,'signed char'),   bytes = 1;
elseif strcmp(type,'uchar')   || strcmp(type,'unsigned char'), bytes = 1;
elseif strcmp(type,'int8')    || strcmp(type,'integer*1'),     bytes = 1;
elseif strcmp(type,'int16')   || strcmp(type,'integer*2'),     bytes = 2;
elseif strcmp(type,'int32')   || strcmp(type,'integer*4'),     bytes = 4;
elseif strcmp(type,'int64')   || strcmp(type,'integer*8'),     bytes = 8;
elseif strcmp(type,'uint8')   || strcmp(type,'integer*1'),     bytes = 1;
elseif strcmp(type,'uint16')  || strcmp(type,'integer*2'),     bytes = 2;
elseif strcmp(type,'uint32')  || strcmp(type,'integer*4'),     bytes = 4;
elseif strcmp(type,'uint64')  || strcmp(type,'integer*8'),     bytes = 8;
elseif strcmp(type,'float32') || strcmp(type,'real*4'),        bytes = 4;
elseif strcmp(type,'float64') || strcmp(type,'real*8'),        bytes = 8;
elseif strcmp(type,'double')  || strcmp(type,'real*8'),        bytes = 8;
else error(['Data type (' type ') unsupported!']);
end;
