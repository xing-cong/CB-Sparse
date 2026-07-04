NVCC := nvcc
NVCC_FLAGS := -std=c++14 -O3 -arch=sm_89 -Xcompiler=-Wno-deprecated-declarations
LDLIBS := -lcudart -lcusparse
TARGET := cusparse_spmv
HEADERS := $(wildcard *.h io/*.h)

.PHONY: all clean

all: $(TARGET)

$(TARGET): main.cu $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) $< -o $@ $(LDLIBS)

clean:
	$(RM) $(TARGET)
