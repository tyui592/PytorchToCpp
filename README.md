# build

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
