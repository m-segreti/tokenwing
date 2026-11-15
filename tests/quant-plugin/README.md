# Test quant plugin

`ext-q4_0.c` is a `--quant-plugin` whose type in slot 0 is byte for byte Q4_0 (ggml's own Q4_0 routines), for checking the
plugin path end to end: a model converted with `make_model_file.py --quant-plugin` must give the same text on the CPU as
the same weights in Q4_0. It is not built with the project; build it after the libraries, from the repository root:

```sh
cc -O2 -shared -fPIC -Iggml/include tests/quant-plugin/ext-q4_0.c -Lbuild/bin -lggml-base -lggml-cpu -Wl,-rpath,$PWD/build/bin -o build/bin/libext-q4_0.so
nvcc -O2 -shared -Xcompiler -fPIC -arch=native -DEXT_Q4_0_CUDA -Iggml/include tests/quant-plugin/ext-q4_0.c tests/quant-plugin/ext-q4_0-cuda.cu -Lbuild/bin -lggml-base -lggml-cpu -Xlinker -rpath,$PWD/build/bin -o build/bin/libext-q4_0-cuda.so
```

`-DEXT_Q4_0_FAIL=N` builds a plugin that is wrong in one way on purpose (the list is at the top of `ext-q4_0.c`), to check
that the loader refuses it with one line.
