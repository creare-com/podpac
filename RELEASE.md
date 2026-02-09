# Release

How to release `podpac`

## Updating main

1. Ensure your local main branch is synced to upstream:

```bash
$ git pull upstream main  # or just git pull
```

2. Create a release branch: `release/4.x.y`

3. Update [`version.py`](podpac/version.py) `MAJOR`, `MINOR`, and `HOTFIX` to the right semantic version

4. Run unit tests

```bash
$ # Activate Python environment
$ pytest podpac             
```

4. Review the [CHANGELOG](CHANGELOG.md) and update
    - Prefix should be a [numpy commit prefix](https://docs.scipy.org/doc/numpy/dev/gitwash/development_workflow.html#writing-the-commit-message)
    - Convention:
```markdown
- <prefix>: <short description> ([<github issue>](https://github.com/creare-com/podpac/issues/<issue#>))
```

5. Merge changes into main 

6. On the main branch, Tag the release:

```bash
$ git tag -a X.Y.Z -m 'X.Y.Z'
```

6. Push your changes to main:

```bash
$ git push upstream main
$ git push upstream --tags
```

7. Build source and binary wheels for pypi (you have to have the `wheels` package installed):

```bash
$ git clean -xdf  # this deletes all uncommited changes!
$ python setup.py bdist_wheel sdist
```

8. Upload package to [TestPypi](https://packaging.python.org/guides/using-testpypi/). You will need to be listed as a package owner at
https://pypi.python.org/pypi/podpac for this to work. You now need to use a pypi generated token, can no longer use your password. 

First create a `${HOME}/.pypirc` file with the below contents, replacing `YOUR_API_TOKEN` with your valid pypi token (file template taken from here https://stackoverflow.com/questions/69065069/http-403-error-while-uploading-python-package-to-pypi):
```
[distutils]
  index-servers =
    pypi
    podpac

[pypi]
  username = __token__
  password = YOUR_API_TOKEN

[YOUR_PROJECT_NAME]
  repository = https://upload.pypi.org/legacy/
  username = __token__
  password = YOUR_API_TOKEN

```

Then run the upload command from your podpac directory

```bash
$ twine upload dist/podpac-X.Y.Z*
```

9. Issue the release announcement (via github)
