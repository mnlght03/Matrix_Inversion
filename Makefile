CC = g++
CFLAGS = -Wall

build_naive:
	$(CC) $(CFLAGS) naiveRealisation/matrix.cpp naiveRealisation/main.cpp -o naive

run_naive: build_naive
	./naive

measure_naive_time:
	$(CC) $(CFLAGS) naiveRealisation/matrix.cpp naiveRealisation/measure_time.cpp -o naive_time && ./naive_time

run_naive_test:
	$(CC) $(CFLAGS) naiveRealisation/matrix.cpp naiveRealisation/matrix_tests.cpp /usr/local/lib/libgtest.a -lpthread -o naive_tests && ./naive_tests

build_blas:
	$(CC) $(CFLAGS) blasRealisation/matrix.cpp -lblas -lcblas blasRealisation/main.cpp -o blas

run_blas: build_blas
	./blas

measure_blas_time:
	$(CC) $(CFLAGS) blasRealisation/matrix.cpp -lblas -lcblas blasRealisation/measure_time.cpp -o blas_time && ./blas_time

run_blas_test:
	$(CC) $(CFLAGS) blasRealisation/matrix.cpp -lblas -lcblas blasRealisation/matrix_tests.cpp /usr/local/lib/libgtest.a -lpthread -o blas_tests && ./blas_tests

build_sse:
	$(CC) $(CFLAGS) sseRealisation/matrix.cpp sseRealisation/main.cpp -o sse

run_sse: build_sse
	./sse

measure_sse_time:
	$(CC) $(CFLAGS) sseRealisation/matrix.cpp sseRealisation/measure_time.cpp -o sse_time && ./sse_time

run_sse_test:
	$(CC) $(CFLAGS) sseRealisation/matrix.cpp sseRealisation/matrix_tests.cpp /usr/local/lib/libgtest.a -lpthread -o sse_tests && ./sse_tests


run_intrinsics_test:
	$(CC) $(CFLAGS) sseRealisation/intrinsics_tests.cpp /usr/local/lib/libgtest.a -lpthread -o intrinsics_tests && ./intrinsics_tests

build_all: build_naive build_sse build_blas

measure_all_time:
	$(CC) $(CFLAGS) naiveRealisation/matrix.cpp naiveRealisation/measure_time.cpp -o naive_time
	$(CC) $(CFLAGS) sseRealisation/matrix.cpp sseRealisation/measure_time.cpp -o sse_time
	$(CC) $(CFLAGS) blasRealisation/matrix.cpp -lblas -lcblas blasRealisation/measure_time.cpp -o blas_time
	./naive_time && ./sse_time && ./blas_time

clean:
	rm naive sse blas intirnsics_tests

rebuild: clean build_all
