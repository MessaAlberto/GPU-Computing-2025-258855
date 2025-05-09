# Compiler and flags
CC := gcc
LIB_FLAGS := -lm
CFLAGS := -g -O3
NVCC := nvcc
NVCC_FLAGS := --gpu-architecture=sm_80 -m64

# Directories
SRC_FOLDER := src
LIB_FOLDER := lib
OBJ_FOLDER := obj
BIN_FOLDER := bin
OUT_FOLDER := output
ERR_FOLDER := output_err

# Files
RUN_SBATCH := script/run.sbatch

MAIN_SRC := $(wildcard $(SRC_FOLDER)/*.c)
MAIN_CU_SRC := $(wildcard $(SRC_FOLDER)/*.cu)

LIB_SRC := $(wildcard $(LIB_FOLDER)/*.c)
LIB_OBJ := $(patsubst $(LIB_FOLDER)/%.c, $(OBJ_FOLDER)/%.o, $(LIB_SRC))

MAIN_BIN := $(patsubst $(SRC_FOLDER)/%.c, $(BIN_FOLDER)/%, $(MAIN_SRC))
MAIN_CU_BIN := $(patsubst $(SRC_FOLDER)/%.cu, $(BIN_FOLDER)/%, $(MAIN_CU_SRC))

.PRECIOUS: $(OBJ_FOLDER)/%.o

# Rules
all: create_dir $(MAIN_BIN) $(MAIN_CU_BIN)

create_dir:
	@mkdir -p $(OBJ_FOLDER) $(BIN_FOLDER) $(OUT_FOLDER) $(ERR_FOLDER)

$(OBJ_FOLDER)/%.o: $(LIB_FOLDER)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BIN_FOLDER)/%: $(SRC_FOLDER)/%.c $(LIB_OBJ)
	$(CC) $(CFLAGS) $< -o $@ $(LIB_OBJ) $(LIB_FLAGS)

$(BIN_FOLDER)/%: $(SRC_FOLDER)/%.cu $(LIB_OBJ)
	module load CUDA/12.1.1 && $(NVCC) $(NVCC_FLAGS) $< -o $@ $(LIB_OBJ) $(LIB_FLAGS)

clean:
	rm -rf $(BIN_FOLDER) $(OUT_FOLDER) $(ERR_FOLDER) $(OBJ_FOLDER)

clean_out:
	rm -rf $(OUT_FOLDER)/* $(ERR_FOLDER)/*
