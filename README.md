C++ Code fro Loading a TorchScript Model
==
This is c++ implementation to load a torchscript [object detector](https://github.com/tyui592/Real_Time_Helmet_Detection).

The detector should take a tensor (1 x Channel(rgb) x Height x Width)  as input and output tuple of 3 tensors (boudning box, class id, confidence score)

Usage
--

### Build

```bash
$ mkdir build
$ cd build
build$ cmake -DCMAKE_PREFIX_PATH=<libtorch_path> ..
build$ cmake --build . --config Release
```

### Run

```bash
build$ ./main -m <model_path> -i <image_path>
```

# References
- https://pytorch.org/docs/stable/jit.html

# TODO
- [ ] Make docker file
- [ ] Video Detection
