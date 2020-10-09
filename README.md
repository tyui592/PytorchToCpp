C++ code to Run the [Object Detector](https://github.com/tyui592/Real_Time_Helmet_Detection)
==

# Build

```bash
$ mkdir build
$ cd build
build$ cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
build$ cmake --build . --config Release
```

# Usage

```bash
build$ ./main -m <model_path> -i <image_path>
```

# TODO
- [ ] Make docker file
- [ ] Video Detection
