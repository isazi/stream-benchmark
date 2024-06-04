ARCH = 80

CC = nvcc
CFLAGS = -ccbin g++ -m64 --x cu -O3 -std=c++17 -I${HOME}/.local/include
CFLAGS += -gencode arch=compute_$(ARCH),code=sm_$(ARCH)
DEFINES = -DUSE_KMM -DKMM_USE_CUDA -DSPDLOG_FMT_EXTERNAL
DEPENDS = ../cuda/stream.cu
LIBS = -L${HOME}/.local/lib -lkmm

stream: stream.cpp $(DEPENDS)
	$(CC) $(CFLAGS) $(DEFINES) -o $@ stream.cpp $(LIBS)

all: stream

clean:
	rm -f stream