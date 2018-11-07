# Development Roadmap

## Overview

PODPAC is in alpha development right now. We are in the middle of a 2-year effort to build this software. The goal for year 1 is to complete all of the core features. In year 2 we plan to build on top of the base functionality to support various earth science applications. 

## Management of Development
We use [Github Projects](https://github.com/creare-com/podpac/projects) to manage development of different PODPAC versions. To get a sense of where this project is going, feel free to have a look, and make suggestions for features in upcoming versions.  

When features / bugs are identified through Github issues, they will be added to the relevant project. The features and bugs will be prioritized, and targeted for a release. In some cases, minor releases will be created to fix important bugs. 

## Versioning scheme

We use the following versioning format: 
`Major.minor.hotfix+hash`
* Major: 
    * For major releases > 1, the interface will remain backwards compatible
    * As an exception to this, the 0.x.x releases (pre-feature-complete) are not guaranteed to be backwards compatibility
* Minor:
    * Each minor release adds requested features, and fixes known bugs
* hotfix:
    * Hotfix releases fix high priority bugs
* +hash: 
    * During development, the git hash is appended to the end of the version
    * This allows a particular point in the development to be referenced
    * Tagged releases will not include this hash

