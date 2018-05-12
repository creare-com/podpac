# Documentation

The following sections outlines how to develop and build the `podpac` documentation.

## Install

- Install **Sphinx** and the **Read the Docs** theme

```bash
$ conda install sphinx              # or `pip install sphinx`
$ conda install sphinx_rtd_theme    # or `pip install sphinx-rtd-theme`
```

- Install `recommonmark` to support markdown input files

```bash
$ pip install recommonmark
```

## Build

### Website

To build the documentation into a website in the `doc/build` directory, run from the `/doc` directory:

```bash
$ make html
or
$ sphinx-build source build   # run sphinx manually to build html by default
$ sphinx-build -aE source build   # for sphinx to rebuild all files (no cache)
```

See [`sphinx-build` docs](http://www.sphinx-doc.org/en/stable/invocation.html#invocation-of-sphinx-build) for more options.

### PDF

To build a pdf from the documentation, you need to install a latex distribution ([MikTex (Windows)](https://miktex.org) [MacTex (Mac)](https://www.tug.org/mactex/)), then run:

```bash
$ make latex                           # build the docs into a tex format in a latex directory
or
$ sphinx-build -b latex source latex   # run sphinx manually to build lated b
```

Enter the build directory and convert tex file to pdf:

```bash
$ cd build                             # go into the latex directory
$ pdflatex WeatherCitizen.tex          # build the pdf from WeatherCitizen.tex entry file
```

## Develop

To live-serve the documentation as a website during development, you will need to add one more python package `sphinx-autobuild`:

```
$ pip install sphinx-autobuild
```

Then run from the `doc` directory:

```
$ sphinx-autobuild -aE source build
```

And the visit the webpage served at http://127.0.0.1:8000. Each time a change to the documentation source is detected, the HTML is rebuilt and the browser automatically reloaded.

To stop the server simply press ^C.

See https://github.com/GaretJax/sphinx-autobuild for more options

## Organization

- `/source` - source documentation files
    + `/source/_templates` - templates to use for styling pages
    + `/source/_static` - static files that need to be copied over to distributed documentation (i.e. images, source code, etc)
    + `/source/conf.py` - sphinx configuration file
    + `/source/index.rst` - root documentation file. Includes TOC
    + ... add others as generated ...
- `/build` - generated documentation files
