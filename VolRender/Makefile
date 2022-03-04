INCLUDES := -I ../inc
LIBS := -lglut -lGL
DEBUG := -g -G
#########################

# Target Rules
all: dir build

build: volRender

volRender: volRender.o volRender_kernel.o
	nvcc $(DEBUG) ./bin/volRender.o $(LIBS) ./bin/volRender_kernel.o -o ./bin/volRender

volRender.o:volRender.cpp
	nvcc $(DEBUG) volRender.cpp $(INCLUDES) -o ./bin/volRender.o -c

volRender_kernel.o:volRender_kernel.cu
	nvcc $(DEBUG) volRender_kernel.cu $(INCLUDES) -o ./bin/volRender_kernel.o -c

dir:
	mkdir -p bin

run: build
	./bin/volRender

clean:
	rm -f ./bin/volRender