all: test

CFLAGS=$(shell pkgconf --cflags openblas64) -DLAPACK_ILP64

LDFLAGS=$(shell pkgconf --libs openblas64 --libs glut --libs opengl)

test: main.c
	gcc -g $(CFLAGS) main.c $(LDFLAGS) -lpthread -o $@
