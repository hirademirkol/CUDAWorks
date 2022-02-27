all: dir build

build: volRender

volRender:
	g++ volRender.cpp -lglut -lGL -o ./bin/volRender

dir:
	mkdir -p bin

run: build
	./bin/volRender

clean:
	rm -f ./bin/volRender