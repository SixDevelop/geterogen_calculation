build/reduce-scan: main.cc linear-algebra.hh
	@mkdir -p build
	g++ -O3 -march=native -fopenmp main.cc -lOpenCL -o build/reduce-scan
build/clang-builder:
	@mkdir -p build
	clang++ -std=c++11 -Xpreprocessor -framework OpenCL main.cc -o build/reduce-scan