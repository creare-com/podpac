@ECHO OFF
ECHO "Fixing hard-coded paths included in local PODPAC conda installation."
REM Write/fix the kernel.json file
set kernels={"argv": ["
set kernele=miniconda\\envs\\podpac\\python.exe",  "-m",  "ipykernel_launcher",  "-f",  "{connection_file}" ], "display_name": "Python 3", "language": "python"}

set mypathescaped=%mypath:\=\\%
set mypathfwd=%mypath:\=/%

del "miniconda\envs\podpac\share\jupyter\kernels\python3\kernel.json"
echo %kernels%%mypathescaped%%kernele% >> miniconda\envs\podpac\share\jupyter\kernels\python3\kernel.json

REM Write/Fix qt.conf

del "miniconda\envs\podpac\qt.conf"
echo [Paths] >> "miniconda\envs\podpac\qt.conf"
echo Prefix = %mypathfwd%miniconda/envs/podpac/Library >> "miniconda\envs\podpac\qt.conf"
echo Binaries = %mypathfwd%miniconda/envs/podpac/Library/bin >> "miniconda\envs\podpac\qt.conf"
echo Libraries = %mypathfwd%miniconda/envs/podpac/Library/lib >> "miniconda\envs\podpac\qt.conf"
echo Headers = %mypathfwd%miniconda/envs/podpac/Library/include/qt >> "miniconda\envs\podpac\qt.conf"

