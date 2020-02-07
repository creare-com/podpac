# AWS

## Public PODPAC distribution

PODPAC is compiled into two distribution .zip archives for building serverless functions.
PODPAC is compile for each version:

- `podpac_dist.zip`: Archive containing PODPAC core distribution
- `podpac_deps.zip`: Archive containing PODPAC dependencies 

These archives are posted publically in the S3 bucket `podpac-dist`.
This bucket has one directory for each podpac version.
The bucket itself is private, but each directory is made public individually.

### Creating new distribution

The following process is used to create new PODPAC distribution in the `podpac-dist` bucket
when a new version of PODPAC is released.

- Run `build_lambda.sh`
- Run `upload_lambda.sh`
- Navigate to `podpac-dist` (or input bucket) and make the archives public
