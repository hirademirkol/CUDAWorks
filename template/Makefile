INCLUDES := -I ../inc
LIBS := -lglut -lGL
DEBUG := -g -G
#########################

# Target Rules
all: dir build

build: template

template: template.o
	nvcc $(DEBUG) $(LIBS) bin/template.o -o bin/template

template.o:template.cpp
	nvcc $(DEBUG) $(LIBS) template.cpp $(INCLUDES) -o bin/template.o -c

dir:
	mkdir -p bin

run: build
	bin/template

clean:
	rm -f bin/template