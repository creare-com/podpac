#!bash
cd ~

# checkout podpac
git checkout https://github.com/creare-com/podpac.git
mv podpac podpac_lib

# Create virtual environment, and install packages
virtualenv env_pip2
source env_pip2/bin/activate
pip install pip --upgrade
pip install affine==2.1.0 boto3==1.4.7 botocore==1.7.19 click==6.7 click-plugins==1.0.3 cligj==0.4.0 docutils==0.14 enum34==1.1.6 futures==3.1.1 jmespath==0.9.3 pyparsing==2.2.0 python-dateutil==2.6.1 rasterio==1.0a8 s3transfer==0.1.11 six==1.11.0 snuggs==1.4.1 Jinja2==2.9.6 MarkupSafe==1.0 Webob==1.7.3 beautifulsoup4==4.6.0 bottleneck==1.2.1 certifi==2017.7.27.1 chardet==3.0.4 dask==0.15.3 decorator==4.1.2 docopt==0.6.2 idna==2.6 ipython-genutils==0.2.0 numexpr==2.6.4 numpy==1.13.1 pandas==0.20.3 pint==0.8.1 pydap==3.2.2 pytz==2017.2 rasterio==1.0a8 requests==2.18.4 scipy==0.19.1 singledispatch==3.4.0.3 traitlets==4.3.2 urllib3==1.22 xarray==0.9.6
deactivate

# Copy python packages from virtual environment for manipulation
mkdir dist
cd dist
cd -r ~/env_pip2/lib/python2.7/dist-packages/* .
cd -r ~/env_pip2/lib64/python2.7/dist-packages/* .
cd -r ~/env_pip2/lib/python2.7/site-packages/* .
cd -r ~/env_pip2/lib64/python2.7/site-packages/* .

# Remove packages already in AWS python
ls /usr/lib/python2.7/dist-packages/ | xargs rm -r
ls /usr/lib64/python2.7/dist-packages/ | xargs rm -r

# Remove the egg-info directories
rm -r *-info*

# Restore Pydap, which is unique
cp -r ~/env_pip2/lib/python2.7/dist-packages/Pydap-3.2.2-py2.7* .
# And a few other packages that pydap needs
cp -r ~/env_pip2/lib/python2.7/dist-packages/beautifulsoup4-4.6.0.dist-info .
cp -r ~/env_pip2/lib/python2.7/dist-packages/docopt-0.6.2-py2.7.egg-info .
cp -r ~/env_pip2/lib/python2.7/dist-packages/Jinja2-2.9.6.dist-info .
cp -r ~/env_pip2/lib/python2.7/dist-packages/singledispatch-3.4.0.3.dist-info .
cp -r ~/env_pip2/lib64/python2.7/dist-packages/WebOb-1.7.3.dist-info .
cp -r ~/env_pip2/lib64/python2.7/dist-packages/numpy-1.13.1.dist-info .
cp -r ~/env_pip2/lib64/python2.7/dist-packages/numpy-1.13.3.dist-info .
cp -r ~/env_pip2/lib64/python2.7/dist-packages/MarkupSafe-1.0-py2.7.egg-info .

# Add __init__.py to pydap, for some reason...
touch pydap/__init__.py
touch pydap/responses/__init__.py
touch pydap/handlers/__init__.py
touch pydap/parsers/__init__.py

# restore certain packages -- I may have fouled the plain AWS installation
cp -r ~/env_pip2/lib/python2.7/dist-packages/requests .
cp -r ~/env_pip2/lib/python2.7/dist-packages/urllib3 .
cp -r ~/env_pip2/lib/python2.7/dist-packages/pkg_resources/ .
cp -r ~/env_pip2/lib/python2.7/dist-packages/jinja2/ .
cp -r ~/env_pip2/lib64/python2.7/dist-packages/markupsafe/ .

# Remove any tests
find . -name "*test*" | grep "testing" -v | grep "_tester" -v | grep "numexp" -v | grep "shortest" -v | xargs rm -r

# Remove any .pyc files
find . -name "*.pyc" | xargs rm 

# Link any repeated .so files together, and delete repeats
cp ~/podpac_lib/dist/aws/link_so.py .
find . -name "*.so" > so_files.txt
python link_so.py
rm so_files.txt
rm link_so.py

# Add the podpac library
cp -r ~/podpac_lib/podpac .
cp -r ~/podpad_lib/dist/aws/handler.py .

# Zip all the directories, and the files in the current directory
find * -maxdepth 0 -type f | grep ".zip" -v | grep -v ".pyc" | xargs zip -9 -rqy podpac_dist.zip
find * -maxdepth 0 -type d -exec zip -9 -rqy {}.zip {} \;

# Figure out the package sizes (for python script)
du -s *.zip > zip_package_sizes.txt
du -s * | grep .zip -v > package_sizes.txt

# Run python script to assemble zip files, and upload to s3
cp ~/podpac_lib/dist/aws/mk_dist.py .
python mk_dist.py 
