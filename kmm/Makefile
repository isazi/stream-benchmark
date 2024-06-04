ARCH = 80

CC = nvcc
CFLAGS = -ccbin g++ -m64 --x cu -O3 -std=c++17
CFLAGS += -gencode arch=compute_$(ARCH),code=sm_$(ARCH)
DEFINES = -DUSE_KMM -DKMM_USE_CUDA
DEPENDS = ../cuda/stream.cu
LIBS = -lkmm

stream: stream.cpp $(DEPENDS)
	$(CC) $(CFLAGS) $(DEFINES) -o $@ stream.cpp $(LIBS)

all: stream

clean:
	rm -f stream