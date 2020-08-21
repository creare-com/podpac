@echo off
call bin\set_local_conda_path.bat
call bin\fix_hardcoded_absolute_paths.bat
call bin\activate_podpac_conda_env.bat

cd podpac
echo "Updating PODPAC"
git fetch
for /f %%a in ('git describe --tags --abbrev^=0 origin/master') do git checkout %%a
cd ..
echo "Updating PODPAC EXAMPLES"
cd podpac-examples
git fetch
for /f %%a in ('git describe --tags --abbrev^=0 origin/master') do git checkout %%a
cd ..
cd podpac
cd dist
echo "Updating CONDA ENVIRONMENT"
conda env update -f windows_conda_environment.yml
cd ..
cd ..



