CC = g++
CFLAGS = -Wall

build_naive:
	$(CC) $(CFLAGS) naiveRealisation/matrix.cpp naiveRealisation/main.cpp -o naive

run_naive: build_naive
	./naive

build_blas:
	$(CC) $(CFLAGS) blasRealisation/matrix.cpp blasRealisation/main.cpp -o blas

run_blas: build_blas
	./blas

build_sse:
	$(CC) $(CFLAGS) sseRealisation/matrix.cpp sseRealisation/main.cpp -o sse

run_sse: build_sse
	./sse

run_intrinsics_test:
	$(CC) $(CFLAGS) sseRealisation/intrinsics_tests.cpp /usr/local/lib/libgtest.a -lpthread -o intrinsics_tests && ./intrinsics_tests

build_all: build_naive build_sse build_blas

clean:
	rm naive sse blas intirnsics_tests

rebuild: clean build_all
