# Release

How to release `podpac`

1. Ensure your master branch is synced to upstream:

```bash
$ git pull upstream master
```

2. Update [`version.py`](podpac/version.py) `MAJOR`, `MINOR`, and `HOTFIX` to the right semantic version

3. Run unit tests for python 2 and python 3 environments

```bash
# python 3 (assumes conda environment is named `podpac`)
$ source activate podpac
$ pytest podpac             

# python 2 (assumes conda environment is named `podpac27`)
$ source activate podpac27
$ pytest podpac
```

4. Review the [CHANGELOG](CHANGELOG.md) and update
    - Prefix should be a [numpy commit prefix](https://docs.scipy.org/doc/numpy/dev/gitwash/development_workflow.html#writing-the-commit-message)
    - Convention:
```markdown
- <prefix>: <short description> ([<github issue>](https://github.com/creare-com/podpac/issues/<issue#>))
```
5. On the master branch, Tag the release:

```bash
$ git tag -a X.Y.Z -m 'X.Y.Z'
```

6. Push your changes to master:

```bash
$ git push upstream master
$ git push upstream --tags
```

7. Build source and binary wheels for pypi:

```bash
$ git clean -xdf  # this deletes all uncommited changes!
$ python setup.py bdist_wheel sdist
```

8. Upload package to [TestPypi](https://packaging.python.org/guides/using-testpypi/). You will need to be listed as a package owner at
https://pypi.python.org/pypi/podpac for this to work.

```bash
$ twine upload --repository-url https://test.pypi.org/legacy/ dist/podpac-X.Y.Z*
```

9. Use twine to register and upload the release on pypi. Be careful, you can't
take this back! You will need to be listed as a package owner at
https://pypi.python.org/pypi/podpac for this to work.

```bash
$ twine upload dist/podpac-X.Y.Z*
```

10. Tag the `master` branch of [creare-com/podpac-examples](https://github.com/creare-com/podpac-examples) with the same semantic version.

11. Issue the release announcement (tbd)
