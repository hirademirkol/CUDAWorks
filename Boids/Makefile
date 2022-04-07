INCLUDES := -I ../inc
LIBS := -lglut -lGL -lGLU
DEBUG := -g -G
BUILDDIR = bin
#########################

# Target Rules
all: $(BUILDDIR) build

build: $(BUILDDIR)/boids

$(BUILDDIR)/boids: $(BUILDDIR)/boids.o $(BUILDDIR)/boidSystem.o $(BUILDDIR)/boidSystem_cuda.o $(BUILDDIR)/renderer.o
	nvcc $(DEBUG) $(LIBS) $(BUILDDIR)/boids.o $(BUILDDIR)/boidSystem.o $(BUILDDIR)/boidSystem_cuda.o $(BUILDDIR)/renderer.o -o $(BUILDDIR)/boids

$(BUILDDIR)/boids.o:boids.cpp
	nvcc $(DEBUG) $(LIBS) boids.cpp $(INCLUDES) -o $(BUILDDIR)/boids.o -c

$(BUILDDIR)/boidSystem.o:boidSystem.cpp
	nvcc $(DEBUG) $(LIBS) boidSystem.cpp $(INCLUDES) -o $(BUILDDIR)/boidSystem.o -c

$(BUILDDIR)/boidSystem_cuda.o:boidSystem_cuda.cu
	nvcc $(DEBUG) $(LIBS) boidSystem_cuda.cu $(INCLUDES) -o $(BUILDDIR)/boidSystem_cuda.o -c

$(BUILDDIR)/renderer.o:renderer.cpp
	nvcc $(DEBUG) $(LIBS) renderer.cpp $(INCLUDES) -o $(BUILDDIR)/renderer.o -c

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

run: build
	$(BUILDDIR)/boids

clean:
	rm -fr $(BUILDDIR)