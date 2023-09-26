all: test-2d

CFLAGS=$(pkgconf --cflags openblas64)
LDFLAGS=$(pkgconf --libs openblas64) -lm

test-2d: main-2d.c
	gcc -O3 $(CFLAGS) main-2d.c $(LDFLAGS) -llapacke -o $@
	./test-2d
	gnuplot test-2d.plt
