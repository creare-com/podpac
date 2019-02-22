@ECHO OFF
ECHO "Setting system path to use local PODPAC conda installation"
SET mypath=%~dp0..\
SET CONDAPATH=%mypath%miniconda;%mypath%miniconda\Library\mingw-w64\bin;%mypath%miniconda\Library\usr\bin;%mypath%miniconda\Library\bin;%mypath%miniconda\Scripts
SET PATH=%CONDAPATH%;%PATH%
SET GDAL_DATA=%mypath%miniconda\envs\podpac\Lib\site-packages\osgeo\data\gdal
SET PYTHONPATH=%mypath%podpac;
SET CURL_CA_BUNDLE=%mypath%miniconda\envs\podpac\Library\ssl\cacert.pem