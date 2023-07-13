# Documentation

The following sections outlines how to develop and build the `podpac` documentation.

## Install

- Install **Sphinx** and the **Read the Docs** theme

```bash
$ pip install sphinx
$ pip install sphinx_rtd_theme
```

- Install `myst-parser` to support markdown input files

```bash
$ pip install myst-parser
```

## Test

To run the doctests in the documentation, run from the `/doc` directory:

```bash
$ ./test-docs.sh        # convienence script

# or

$ sphinx-build -b doctest source build  # run tests manually
```

## Build

### Website

To build the documentation into a website in the `doc/build` directory, run from the `/doc` directory:

```bash
$ ./release-docs.sh     # convienence script

# or

$ sphinx-build source build       # run manually
$ sphinx-build -aE source build   # rebuild all files (no cache)
```

See [`sphinx-build` docs](http://www.sphinx-doc.org/en/stable/invocation.html#invocation-of-sphinx-build) for more options.

### PDF

To build a pdf from the documentation, you need to install a latex distribution ([MikTex (Windows)](https://miktex.org) [MacTex (Mac)](https://www.tug.org/mactex/)), then run:

```bash
$ make latex                           # build the docs into a tex format in a latex directory

# or

$ sphinx-build -b latex source latex   # run sphinx manually to build latex
```

Enter the build directory and convert tex file to pdf:

```bash
$ cd build                     # go into the latex directory
$ pdflatex podpac.tex          # build the pdf from podpac.tex entry file
```

## Develop

To live-serve the documentation as a website during development, you will need to add one more python package [`sphinx-autobuild`](https://github.com/GaretJax/sphinx-autobuild):

```bash
$ pip install sphinx-autobuild
```

Then run from the `doc` directory:

```bash
$ ./serve-docs.sh                   # convienence script

# or

$ sphinx-autobuild source build     # run manually
$ sphinx-autobuild -aE source build # rebuild all files (no cache)
```

And the visit the webpage served at `http://127.0.0.1:8000`. Each time a change to the documentation source is detected, the HTML is rebuilt and the browser automatically reloaded.

To stop the server simply press `^C`.

## Organization

- `/source` - source documentation files
    + `/source/_templates` - templates to use for styling pages
    + `/source/_static` - static files that need to be copied over to distributed documentation (i.e. images, source code, etc)
    + `/source/conf.py` - sphinx configuration file
    + `/source/index.rst` - root documentation file
    + `/source/api/` - auto generated API documentation using `sphinx-autogen`
- `/build` - generated documentation files

## References

- [Sphinx docstring interpretation (autodoc)](http://www.sphinx-doc.org/en/stable/ext/autodoc.html#module-sphinx.ext.autodoc)
- [Sphinx Themes](http://www.sphinx-doc.org/en/stable/theming.html)
    + [Read the Docs Theme](https://github.com/rtfd/sphinx_rtd_theme)
- [Sphinx doctest](http://www.sphinx-doc.org/en/master/ext/doctest.html)
