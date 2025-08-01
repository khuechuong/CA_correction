# CA_correction
Chromatic Abberation Correction

A speed optimized version of [Chromatic_aberration_correction](https://github.com/RayXie29/Chromatic_aberration_correction). 

Chromatic aberration correction completed in approximate 18 ms for a 2.8 MP (1936×1464) image.

## Instruction
```
cd CA_correction

g++ ca_correction.cpp -o ca_correction `pkg-config --cflags --libs opencv4` -ffast-math -fopenmp -O3

./ca_correction <path to input image> <path to output image>
```
