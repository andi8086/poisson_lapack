all: test

CFLAGS=$(pkgconf --cflags openblas64)
LDFLAGS=$(pkgconf --libs openblas64)

test: main.c
	gcc $(CFLAGS) main.c $(LDFLAGS) -llapacke -o $@
	./test
