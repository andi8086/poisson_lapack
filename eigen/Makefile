all: test

CFLAGS=$(shell pkgconf --cflags openblas64)
LDFLAGS=$(shell pkgconf --libs openblas64)

CFLAGS += -DLAPACK_ILP64

test: main.c
	gcc -O3 $(CFLAGS) main.c $(LDFLAGS) -o $@
