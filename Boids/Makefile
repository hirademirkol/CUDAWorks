INCLUDES := -I ../inc
LIBS := -lglut -lGL -lGLU
DEBUG := -g -G
#########################

# Target Rules
all: dir build

build: boids

boids: boids.o boidSystem.o renderer.o
	nvcc $(DEBUG) $(LIBS) bin/boids.o bin/boidSystem.o bin/renderer.o -o bin/boids

boids.o:boids.cpp
	nvcc $(DEBUG) $(LIBS) boids.cpp $(INCLUDES) -o bin/boids.o -c

boidSystem.o:boidSystem.cpp
	nvcc $(DEBUG) $(LIBS) boidSystem.cpp $(INCLUDES) -o bin/boidSystem.o -c

renderer.o:renderer.cpp
	nvcc $(DEBUG) $(LIBS) renderer.cpp $(INCLUDES) -o bin/renderer.o -c

dir:
	mkdir -p bin

run: build
	bin/boids

clean:
	rm -f bin/boids