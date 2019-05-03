@ECHO OFF
ECHO "Activating PODPAC environment."
REM This assumes that set_local_conda_path.bat has been called
SET CURL_CA_BUNDLE=%mypath%miniconda\envs\podpac\Library\ssl\cacert.pem
activate podpac
