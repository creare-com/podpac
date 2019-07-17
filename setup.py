"""
podpac module
"""

# Always perfer setuptools over distutils
import sys
import subprocess

from setuptools import find_packages, setup
from setuptools.command.develop import develop

# get version
sys.path.insert(0, "podpac")
import version

__version__ = version.version()

install_requires = [
    "matplotlib>=2.1",
    "numpy>=1.14",
    "pint>=0.8",
    "scipy>=1.0",
    "traitlets>=4.3",
    "xarray>=0.10",
    "requests>=2.18",
    "pyproj>=2.2",
    "lazy-import>=0.2.2",
    "psutil",
]

if sys.version_info.major == 2:
    install_requires += ["future>=0.16"]

extras_require = {
    "datatype": [
        "beautifulsoup4>=4.6",
        "h5py>=2.9",
        "lxml>=4.2",
        "pydap>=3.2",
        "rasterio>=1.0",
        "zarr>=2.3"
    ],
    "aws": [
        "awscli>=1.11",
        "boto3>=1.4",
        "s3fs>=0.2"
    ],
    "algorithms": [
        "numexpr>=2.6"
    ],
    "notebook": [
        "jupyterlab",
        "ipyleaflet",
        "ipywidgets",
        "ipympl",
        "nodejs",
        #'cartopy'
    ],
    "esri": [
        # 'arcpy',
        "certifi>=2018.1.18",
        "urllib3>=1.22",
    ],
    "dev": [
        "pylint>=1.8.2",
        "pytest>=3.3.2",
        "pytest-cov>=2.5.1",
        "pytest-html>=1.7.0",
        "recommonmark>=0.4",
        "sphinx>=1.8",
        "sphinx-rtd-theme>=0.4",
        "sphinx-autobuild>=0.7",
        "coveralls>=1.3",
        "six>=1.0",
        "attrs>=17.4.0",
        "pre_commit>=1"
    ],
}

if sys.version => 3.6:
    extras_require["dev"] += [
        "black",
    ]

# set long description to readme
with open("README.MD") as f:
    long_description = f.read()

all_reqs = []
for key, val in extras_require.items():
    if "key" == "dev":
        continue
    all_reqs += val
extras_require["all"] = all_reqs
extras_require["devall"] = all_reqs + extras_require["dev"]

# install pre-commit hooks after setup in develop mode
class PostDevelopCommand(develop):
    def run(self):
        subprocess.check_call(["pre-commit", "install"])
        develop.run(self)

setup(
    # ext_modules=None,
    name="podpac",
    version=__version__,
    description="Pipeline for Observational Data Processing, Analysis, and Collaboration",
    author="Creare",
    url="https://podpac.org",
    license="APACHE 2.0",
    classifiers=[
        # How mature is this project? Common values are
        # 3 - Alpha
        # 4 - Beta
        # 5 - Production/Stable
        "Development Status :: 4 - Beta",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: GIS",
        # Pick your license as you wish (should match "license" above)
        "License :: OSI Approved :: Apache Software License",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python",
    ],
    packages=find_packages(),
    install_requires=install_requires,
    extras_require=extras_require,
    cmdclass={"develop": PostDevelopCommand},
    long_description=long_description,
    long_description_content_type="text/markdown"
    # entry_points = {
    #     'console_scripts' : []
    # }
)