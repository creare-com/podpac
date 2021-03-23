@ECHO OFF
ECHO "Launching PODPAC Jupyter Lab notebooks."
call bin\set_local_conda_path.bat
call bin\fix_hardcoded_absolute_paths.bat
call bin\activate_podpac_conda_env.bat

cd podpac-examples
jupyter lab 
cd ..
