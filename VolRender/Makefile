INCLUDES := -I ../inc

#########################

# Target Rules
all: dir build

build: volRender

volRender: volRender.o volRender_kernel.o
	nvcc ./bin/volRender.o -lglut -lGL ./bin/volRender_kernel.o -o ./bin/volRender

volRender.o:volRender.cpp
	nvcc volRender.cpp $(INCLUDES) -o ./bin/volRender.o -c

volRender_kernel.o:volRender_kernel.cu
	nvcc volRender_kernel.cu $(INCLUDES) -o ./bin/volRender_kernel.o -c

dir:
	mkdir -p bin

run: build
	./bin/volRender

clean:
	rm -f ./bin/volRender