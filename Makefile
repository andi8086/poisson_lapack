all: test-1d test-2d

CFLAGS=$(pkgconf --cflags openblas64)
LDFLAGS=$(pkgconf --libs openblas64)

test-1d: main-1d.c
	gcc $(CFLAGS) main-1d.c $(LDFLAGS) -llapacke -o $@
	./test-1d > test.dat
	gnuplot test-1d.plt

test-2d: main-2d.c
	gcc $(CFLAGS) main-2d.c $(LDFLAGS) -llapacke -o $@
	./test-2d > test2d.dat
	gnuplot test-2d.plt
