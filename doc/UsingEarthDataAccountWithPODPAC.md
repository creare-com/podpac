# Using Earth Data Account with PODPAC
## Motivation
* An Earth Data Login account is needed to access the wealth of data provided by 
NASA. 
* PODPAC automatically retrieves this data using OpenDAP. 
* OpenDAP needs an authenticated session to retrieve the data. 
* To set up an authenticated session, a user can:
    * save their credentials as part of a PODPAC settings files
    * provide their username and password to the child class of a PyDAP node at runtime
   
## Creating an EarthData Login Account
* Go to the [EarthData Registration Page](https://urs.earthdata.nasa.gov/users/new)
  page and follow the instructions to register an account
* Go to the [EarthData Login Page](https://urs.earthdata.nasa.gov/) to log into
  your account
* To enable OpenDAP access: 
    * Go to your [Profile](https://urs.earthdata.nasa.gov/profile) once logged in.
    * Click on `My Applications`
    * Scroll to the bottom and click on `APPROVE MORE APPLICATIONS`
    * Find the `NASA GESDICS DATA ARCHIVE` and `NSIDC V0 OPeNDAP` applications
        * Additional applications may be required to access datasets of interest
    * For each, click the `APPROVE` button
* At this stage, your EarthData account should be set up and ready to use for
  accessing data through OpenDAP
    
## Saving EarthData Credentials in PODPAC Settings
For convenience, PODPAC can store your EarthData login details so that you do
not have to provide your username and password at run time or in every script. 

> **NOTE:** PODPAC stores your credentials in a plain text file. If this is
a security issue for you, do not use this method. 

To store your credentials use the following code in an interactive Python session: 
```python
from podpac.core.authentication import EarthDataSession
eds = EarthDataSession()
eds.update_login()
```
Then follow the on-screen prompts to enter our username and password. 

Your credentials will be saved in `$HOME\.podpac\settings.json`
where `$HOME` is usually `C:\users\<USERNAME>` on Windows and `/home/<USERNAME>`
on Linux systems. 

## Setting credentials at Runtime
To set credentials at runtime, you can either provide an authenticated session
or the username and password to the PyDAP node or child node. For example

```python
from podpac.core.authentication import EarthDataSession
eds = EarthDataSession(username=<username>, password=<password>)
from podpac.core.data.type import PyDAP
pydap_node = PyDAP(source=<opendap_url>, auth_session=eds)
```

Or 

```python
from podpac.core.data.type import PyDAP
pydap_node = PyDAP(source=<opendap_url>, username=<username>, password=<password>)
```