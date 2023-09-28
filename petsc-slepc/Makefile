all: test

export PKG_CONFIG_PATH=$(SLEPC_DIR)/lib/pkgconfig

CFLAGS += $(shell pkgconf --cflags slepc)
LDFLAGS += $(shell pkgconf --libs slepc)

LDFLAGS += -lmpi

test: main.c
	gcc $^ -o $@ $(CFLAGS) $(LDFLAGS)
