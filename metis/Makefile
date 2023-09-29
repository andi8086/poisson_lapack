all: test


test: main.c
	gcc -o $@ $^ $(CFLAGS) $(LDFLAGS) -lmetis -lgmsh
