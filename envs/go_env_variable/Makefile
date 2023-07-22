CC := g++
SRC1 = ./cpp_src/*.cc
TAR1 = ./go_env_var.dll

all: go_env_var

go_env_var:
	$(CC) $(SRC1) -std=gnu++11 -shared -fPIC -o $(TAR1)

clean:
	rm -rf $(TAR1)