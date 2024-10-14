include ../include/Makefile.inc

SOURCE_CU   = $(shell find . -name '*.cu' 2>/dev/null)
SOURCE_PY   = $(shell find . -name '*.py' 2>/dev/null)
OBJ         = $(shell find . -name *.o 2>/dev/null)
DEP         = $(OBJ:.o=.d)
TARGET_SO   = $(SOURCE_CU:.cu=.so)

-include $(DEP)

all: $(TARGET_SO)

%.so: %.o
	$(NVCC) $(SOFLAG) $(LDFLAG) -o $@ $+
# nvcc是指定编译器，-shared是指定生成动态链接库，-o是指定生成的动态链接库的名字，$+是指定生成动态链接库的目标文件


%.o: %.cu
	$(NVCC) $(CUFLAG) $(INCLUDE) -M -MT $@ -o $(@:.o=.d) $<
	$(NVCC) $(CUFLAG) $(INCLUDE) -o $@ -c $<

.PHONY: test
# PHONY是一个伪目标，它表示不管是否存在这个文件，只要执行这个目标，就会执行后面的命令
# 伪目标是指不生成任何文件，只是执行一些特定的命令
test:
	make clean
	make
	python3 $(SOURCE_PY)

.PHONY: clean
clean:
	rm -rf ./*.d ./*.o ./*.so ./*.exe ./*.plan