all: test

CFLAGS=$(shell pkgconf --cflags openblas)

LDFLAGS=$(shell pkgconf --libs openblas)

test: main.c
	gcc -g $(CFLAGS) main.c $(LDFLAGS) -o $@
