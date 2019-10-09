# CUDA ProCell

[CUDA](http://www.nvidia.it/object/cuda-parallel-computing-it.html) version of 
[ProCell](https://github.com/aresio/ProCell) software.

## Installation

Software requirements:

- [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) `>= 8.0`

- [CMake](https://cmake.org/) `>= 3.8`

```sh
$ git clone https://github.com/ericniso/cuda-pro-cell.git

$ cd cuda-pro-cell

$ mkdir build

$ cd build

$ cmake -DCMAKE_CUDA_FLAGS="-arch=${GPU compute capability}" ..

$ make
```

The Compute Capability of your GPU can be found at this link 
[CUDA GPU compute capabilities list](https://developer.nvidia.com/cuda-gpus).

It must be `>= 3.5` in order to compile and execute the program.

For example, if you plan to use a `GTX 970` with compute capability `5.2`,
the cmake command would look like the following:

```sh
$ cmake -DCMAKE_CUDA_FLAGS="-arch=sm_52" ..
```

## Usage

The generated executable file is located under `build/bin/` as `procell`

- `-h` or `--histogram` is the path of a file containing the histogram of 
    the starting frequency values of the cell fluorescences:

    ```txt
    // histogram.txt

    <fluorescence value> <frequency>
    .
    .
    .
    <fluorescence value> <frequency>
    ```

    For example:

    ```txt
    1.0 0
    8.144 53
    .
    .
    .
    9823.85 274
    ```

- `-c` or `--cell-types` is the path of a file containing the list of 
    subpopulation types properties:

    ```txt
    <ratio> <mean> <stddev>
    ```

    For example:

    ```txt
    // types.txt

    0.53 48.33 21.6
    0.29 86.3 26.8
    ```

    Quiescent type uses `-1` as `mean` and `stddev`:

    ```txt
    // types-with-quiescent.txt

    0.18 -1 -1
    ```

- `-t` or `--time-max` is a `double` value `>= 0` which specifies the max 
    simulation time for cell divisions.
        
- `-o` or `--output` is the path of a file which will be used to store the 
    resulting histogram values in the same format as the starting histogram.
    
- [OPTIONAL] `-p` or `--phi` is a `double` value `> 0` which specifies the 
    minimum fluorescence threshold for cell proliferation.

    If not provided, the minimum `fluorescence value` with `frequency > 0` from 
    the initial histogram will be used.

- [OPTIONAL] `-r` or `--track-ratio` tells the software to keep track of each 
    input cell type frequency for each final fluorescence value.

    The ouput file is modified as follows:

    ```txt
    <fluorescence value> <total frequency> <cell type 1 frequency> ... <cell type nth frequency>
    .
    .
    .
    <fluorescence value> <total frequency> <cell type 1 frequency> ... <cell type nth frequency>
    ```

    The cell type frequencies are ordered as they are specified in the `-c` input file.
