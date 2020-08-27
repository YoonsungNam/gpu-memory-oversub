#!/bin/sh

set -e
   
flex brew.l
g++ lex.yy.c -lfl

./a.out < $1 > ${1%.cu}-mod.cu
#compile generated code

nvcc ${1%.cu}-mod.cu -o ${1%.cu}-mod -I../../common/inc
