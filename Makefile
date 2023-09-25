all: test-1d test-2d

test-1d: main-1d.c
	gcc -llapack -llapacke main-1d.c -o $@
	./test-1d > test.dat
	gnuplot test-1d.plt

test-2d: main-2d.c
	gcc -llapack -llapacke main-2d.c -o $@
	./test-2d > test2d.dat
	gnuplot test-2d.plt
