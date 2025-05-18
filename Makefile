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
MAIN_SRC := $(SRC_FOLDER)/main.c
KERNEL_SRC := $(SRC_FOLDER)/kernel.cu

LIB_SRC := $(wildcard $(LIB_FOLDER)/*.c)
LIB_OBJ := $(patsubst $(LIB_FOLDER)/%.c, $(OBJ_FOLDER)/%.o, $(LIB_SRC))

KERNEL_VERSIONS := 1 2 3 4

MAIN_BIN := $(BIN_FOLDER)/main
OPT_MAIN_BIN := $(BIN_FOLDER)/opt_main
KERNEL_BIN := $(addprefix $(BIN_FOLDER)/kernel_v, $(KERNEL_VERSIONS))

.PRECIOUS: $(OBJ_FOLDER)/%.o

# Rules
all: chmod create_dir $(MAIN_BIN) $(OPT_MAIN_BIN) $(KERNEL_BIN)

chmod:
	@chmod +x $(wildcard script/*)

create_dir:
	@mkdir -p $(OBJ_FOLDER) $(BIN_FOLDER) $(OUT_FOLDER) $(ERR_FOLDER)

$(OBJ_FOLDER)/%.o: $(LIB_FOLDER)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(MAIN_BIN): $(MAIN_SRC) $(LIB_OBJ)
	$(CC) $(CFLAGS) $< -o $@ $(LIB_OBJ) $(LIB_FLAGS)

$(OPT_MAIN_BIN): $(MAIN_SRC) $(LIB_OBJ)
	$(CC) $(OPT_CFLAGS) $< -o $@ $(LIB_OBJ) $(LIB_FLAGS)

$(BIN_FOLDER)/kernel_v%: $(KERNEL_SRC) $(LIB_OBJ)
	module load CUDA/12.1.1 && \
	$(NVCC) $(NVCC_FLAGS) -DSELECT_KERNEL=$* $< -o $@ $(LIB_OBJ) $(LIB_FLAGS)

download:
	mkdir -p $(MTX_FOLDER)
	./script/download_mtx.sh

test:
	./script/submit_all.sh

clean:
	rm -rf $(BIN_FOLDER) $(OUT_FOLDER) $(ERR_FOLDER) $(OBJ_FOLDER)

clean_out:
	rm -rf $(OUT_FOLDER)/* $(ERR_FOLDER)/*

clean_mtx:
	rm -rf $(MTX_FOLDER)/*
