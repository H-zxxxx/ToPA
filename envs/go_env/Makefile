CC := g++
SRC1 = ./cpp_src/*.cc
TAR1 = ./go_env9.so
TAR2 = ./go_env19.so

all: go_env9 go_env19

go_env9:
	$(CC) $(SRC1) -std=gnu++11 -shared -fPIC -D BOARD9x9 -o $(TAR1)

go_env19:
	$(CC) $(SRC1) -std=gnu++11 -shared -fPIC -o $(TAR2)

clean:
	rm -rf $(TAR1) $(TAR2)