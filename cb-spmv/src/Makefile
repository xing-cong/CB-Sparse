NVCC ?= nvcc
NVCC_FLAGS := -O3 -w -arch=sm_89

OPTIONS := -Xcompiler -fopenmp

make:
	$(NVCC) $(NVCC_FLAGS) $(OPTIONS) main.cu -o cb-spmv

clean:
	rm -rf cb-spmv
