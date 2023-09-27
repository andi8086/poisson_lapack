all: test

CFLAGS=$(shell pkgconf --cflags openblas64) -DLAPACK_ILP64

LDFLAGS=$(shell pkgconf --libs openblas64)

test: main.c
	gcc -g $(CFLAGS) main.c $(LDFLAGS) -o $@
