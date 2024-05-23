# Release

How to release `podpac`

## Branch Notes
There are currently two actively maintained versions of PODPAC:
* 3.x version which uses main-3.X and develop-3.X
* 4.x version which uses main and develop

The 3.x is maintained for projects that have not migrated to the new 4.x interface. 

The repercussions of this are as follows:
* Most of the code bases are still compatible, so any new features or bug fixes can go to both branches
* Process is: Feature or hotfix branch off develop-3.X, then when done coding, merge into both develop-3.X and develop

## Updating main

1. Ensure your local main / main-3.X branch is synced to upstream:

```bash
$ git pull upstream main  # or just git pull
```

2. Create a release branch: `release/3.x.y`

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

5. Merge changes into main / main-3.X branch

6. On the main branch, Tag the release:

```bash
$ git tag -a X.Y.Z -m 'X.Y.Z'
```

6. Push your changes to main / main-3.X:

```bash
$ git push upstream main
$ # OR 
$ git push upstream main-3.X
$ git push upstream --tags
```

7. Build source and binary wheels for pypi (you have to have the `wheels` package installed):

```bash
$ git clean -xdf  # this deletes all uncommited changes!
$ python setup.py bdist_wheel sdist
```

8. Upload package to [TestPypi](https://packaging.python.org/guides/using-testpypi/). You will need to be listed as a package owner at
https://pypi.python.org/pypi/podpac for this to work. You now need to use a pypi generated token, can no longer use your password. 

```bash
$ twine upload --repository-url https://test.pypi.org/legacy/ dist/podpac-X.Y.Z*
```

9. Use twine to register and upload the release on pypi. Be careful, you can't
take this back! You will need to be listed as a package owner at
https://pypi.python.org/pypi/podpac for this to work.

```bash
$ twine upload dist/podpac-X.Y.Z*
```

11. Issue the release announcement (via github)
