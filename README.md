# Leveraging scale- and orientation covariant features for planar motion estimation
C++ code for the ECCV 2024 paper

```
@inproceedings{valtonen-ornhag-etal-eccv-2024,
  author       = {Marcus {Valtonen~{\"O}rnhag} and Alberto Jaeval G\'{a}lvez},
  title        = {Leveraging scale- and orientation covariant features for planar motion estimation},
  booktitle    = {European Conference on Computer Vision (ECCV)},
  year         = {2024},
}
```

Please cite the paper if you use the code in academic publications.

## Installation
This project uses CMake, which needs to be available to compile it

For Ubuntu

```console
sudo apt-get install cmake libeigen3-dev
```
Alternatively, there is a Docker file in ``docker``.

## Building
Then you may use the build script

```console
./build.sh
```

This generates an executable in the ``_build`` folder which you can execute.

## Running the code
Run ``benchmark`` in the ``_build`` folder. Here is an example output

```console
$ ./_build/benchmark 
Running: 100000 times
Mean execution time (Guan et al. (LS)): 21686 ns
Mean execution time (solver_valtonen_ornhag_eccv_2024): 348 ns
Mean execution time (solver_choi_kim_2018 2pt): 453 ns
Mean execution time (Guan et al. (CS)): 1387 ns
```

Note that the timing may differ on your hardware.
