all: test

CFLAGS += $(shell pkgconf --cflags slepc)
LDFLAGS += $(shell pkgconf --libs slepc)

LDFLAGS += -lmpi

test: main.c
	gcc $^ -o $@ $(CFLAGS) $(LDFLAGS)
