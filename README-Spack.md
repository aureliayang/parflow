# Building ParFlow with Spack

This document covers building ParFlow with Spack.  This feature is
expiremental and under development.  Building with Spack may be useful
in some situations since Spack builds a complete dependency tree and
thus may have fewer compliling and linking issues.  Spack is a
multi-platform package manager that builds and installs multiple
versions and configurations of software. Spack has support for Linux,
macOS, and many supercomputers.

See [Spack documentation](https://spack.io) for additional information
on Spack and how to use it.

## Obtaining and installing Spack

A special version of Spack is currently needed for a ParFlow build.
There are only two small changes to Spack to address a Silo build
issue and adding the rules for the ParFlow configuration.  Once the
ParFlow build is stable we will ask for ParFlow to be added to the
standard Spack distribution.

The following will download Spack and setup the environment.  The
source command needs to be done in every new shell to set setup the
envirnment.

```shell
   git clone git@github.com:parflow/spack.git
   source spack/share/spack/setup-env.sh
```   

## Building ParFlow

First download, configure, and build ParFlow and all of the libraries
ParFlow depends on.  Spack does all of this for you!  This step will
take a considerable amount of time to be prepared to wait.  You must
be connected to the internet so Spack can download everything.  You
must have a compiler suite (C/C++/Fortran) installed.

```shell
   spack install parflow@develop
```   

Hopefully everything built correctly.

## Running the Spack build of ParFlow 

First setup some environment variables for running out of the Spack
directories.  These use the Spack location command to find the
location of TCL, OpenMPI and ParFlow.  The bin directories are added
to the path.   

```shell
  export PARFLOW_DIR=$(spack location --install-dir parflow)
  export PARFLOW_TCL_DIR=$(spack location --install-dir tcl)
  export PARFLOW_MPI_DIR=$(spack location --install-dir openmpi)
  export PATH=${PARFLOW_MPI_DIR}/bin:$${PARFLOW_TCL_DIR}/bin:${PARFLOW_DIR}/bin:$PATH

  # Are we using spack versions?
  echo "Using PARFLOW_DIR=${PARFLOW_DIR}"
  echo "Using tclsh : $(which tclsh)"
  echo "Using mpirun : $(which mpirun)"
```
We are looking at how to make this installation simpler; possibly using a Spack view.

## Testing 

Spack is managing the directories for the ParFlow build so we checkout
the ParFlow repo to get a copy of the ParFlow regression tests so we
can run them.

Clone the ParFlow repository:

```shell
git clone git@github.com:parflow/parflow.git
```

Run a few tests:

```shell
  cd parflow/test

  tclsh default_single.tcl 1 1 1

  tclsh default_single.tcl 2 2 2

  tclsh default_richards_with_netcdf.tcl 1 1 1

  tclsh default_richards_with_netcdf.tcl 2 2 2
```

Hopefully these all pass!

To run the full test suite:

```shell
  cd parflow/test
  make test
```

Some of the tests that require a large number of MPI ranks may fail
(The error message is something like "There are not enough slots
available in the system to satisfy the 27 slots") if you don't have
enough cores.  The test suite goes up to 27 cores.  We are working to
fix this issue (for MPI folks, overcommiting is not being enabled or
working in the Spack OpenMPI setup).