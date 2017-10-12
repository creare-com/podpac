# Always perfer setuptools over distutils
from setuptools import setup, find_packages, Extension

setup(
    # ext_modules=None,
    name='podpac',

    version='0.0.0',

    description="Pipeline for Observational Data Processing, Analysis, and Collaboration",
    author='MPU',
    # url="https://github.com/creare-com/podpac",
    license="APACHE 2.0",
    classifiers=[
        # How mature is this project? Common values are
        # 3 - Alpha
        # 4 - Beta
        # 5 - Production/Stable
        'Development Status :: 3 - Alpha',
        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: GIS',
        # Pick your license as you wish (should match "license" above)
         'License :: OSI Approved :: Apache Software License',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: both',
    ],
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'xarray',
        'traitlets',
        'pint',
        'numerexpr',
        # Optional requirements
        'rasterio>=1.0',
        'pydap',
        'requests', 
        'beautifulsoup4',   
        'lxml',
        ],
    # entry_points = {
    #     'console_scripts' : []
    # }
)
