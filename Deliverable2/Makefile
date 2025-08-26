# Compiler and flags
CC := gcc
LIB_FLAGS := -lm
CFLAGS := -g -O3 -march=native
OPT_CFLAGS := $(CFLAGS) -DUSE_OPTIM
NVCC := nvcc
NVCC_FLAGS := --gpu-architecture=sm_80 -m64 -O3

# Directories
SRC_FOLDER := src
LIB_FOLDER := lib
OBJ_FOLDER := obj
BIN_FOLDER := bin
OUT_FOLDER := output
ERR_FOLDER := output_err
MTX_FOLDER := mtx

# Files
CUPARSE_SRC := $(SRC_FOLDER)/cuparse.cu
CUDA_SRCS := $(filter-out $(CUPARSE_SRC), $(wildcard $(SRC_FOLDER)/*.cu))

LIB_SRC := $(wildcard $(LIB_FOLDER)/*.c)
LIB_OBJ := $(patsubst $(LIB_FOLDER)/%.c, $(OBJ_FOLDER)/%.o, $(LIB_SRC))

CUDA_BINS := $(patsubst $(SRC_FOLDER)/%.cu, $(BIN_FOLDER)/%, $(CUDA_SRCS))
CUPARSE_BIN := $(BIN_FOLDER)/cuparse

.PRECIOUS: $(OBJ_FOLDER)/%.o

# Rules
all: chmod create_dir $(CUDA_BINS) $(CUPARSE_BIN)

chmod:
	@chmod +x $(wildcard script/*)

create_dir:
	@mkdir -p $(OBJ_FOLDER) $(BIN_FOLDER) $(OUT_FOLDER) $(ERR_FOLDER)

$(OBJ_FOLDER)/%.o: $(LIB_FOLDER)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BIN_FOLDER)/%: $(SRC_FOLDER)/%.cu $(LIB_OBJ)
	module load CUDA/12.1.1 && \
	$(NVCC) $(NVCC_FLAGS) $< -o $@ $(LIB_OBJ) $(LIB_FLAGS)

$(CUPARSE_BIN): $(CUPARSE_SRC) $(LIB_OBJ)
	module load CUDA/12.1.1 && \
	$(NVCC) $(NVCC_FLAGS) $< -o $@ $(LIB_OBJ) $(LIB_FLAGS) -lcusparse

download:
	mkdir -p $(MTX_FOLDER)
	./script/download_mtx.sh

test:
	./script/submit_all.sh

test_block:
	./script/submit_blocksize.sh

clean:
	rm -rf $(BIN_FOLDER) $(OUT_FOLDER) $(ERR_FOLDER) $(OBJ_FOLDER)

clean_out:
	rm -rf $(OUT_FOLDER)/* $(ERR_FOLDER)/*

clean_mtx:
	rm -rf $(MTX_FOLDER)/*
