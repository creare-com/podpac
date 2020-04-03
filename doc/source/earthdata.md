# NASA Earth Data Login

This document describes using an Earth Data Account with PODPAC.

## Motivation

* An Earth Data Login account is needed to access the wealth of data provided by 
NASA. 
* PODPAC automatically retrieves this data using OpenDAP. 
* OpenDAP needs an authenticated session to retrieve the data. 
* To set up an authenticated session, a user can:
    * save their credentials as part of a PODPAC settings files
    * provide their username and password to the child class of a PyDAP node at runtime
   
## Creating an Earthdata Login Account

* Go to the [Earthdata Registration Page](https://urs.earthdata.nasa.gov/users/new)
  page and follow the instructions to register an account
* Go to the [Earthdata Login Page](https://urs.earthdata.nasa.gov/) to log into
  your account
* To enable OpenDAP access: 
    * Go to your [Profile](https://urs.earthdata.nasa.gov/profile) once logged in.
    * Under `Applications`, click on `Authorized Apps`
    * Scroll to the bottom and click on `APPROVE MORE APPLICATIONS`
    * Find the `NASA GESDICS DATA ARCHIVE`, `NSIDC V0 OPeNDAP`, and `NSIDC_DATAPOOL_OPS` applications
        * Additional applications may be required to access datasets of interest
    * For each, click the `APPROVE` button
* At this stage, your Earthdata account should be set up and ready to use for
  accessing data through OpenDAP
    
## Using Earthdata Credentials in PODPAC


PODPAC uses Earthdata credentials to access the SMAP data source nodes.
You can store the credentials for SMAP nodes using the `Node` method `set_credentials`.

To store credentials for SMAP nodes, use the following code in an interactive Python session: 

```python
from podpac.datalib import SMAP

node = SMAP()
node.set_credentials(username="<earthdata-username>", password="<earthdata-password>")
```

The `set_credentials` method stores credentials for a Node in the PODPAC settings.
To persistently save the credentials in the PODPAC settings
(to avoid running `set_credentials` at runtime or in a script), run `settings.save()`:

> **NOTE:** PODPAC stores credentials in plain text.
> Be conscious of outputting the PODPAC settings to a file when it contains credentials.

```
from podpac import settings

settings.save()
```

Your credentials will be saved in `$HOME\.podpac\settings.json`
where `$HOME` is usually `C:\users\<USERNAME>` on Windows and `/home/<USERNAME>`
on Linux systems.
