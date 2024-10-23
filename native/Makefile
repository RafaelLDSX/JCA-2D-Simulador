#####################################################################
#
# Source: 
#         https://www3.ntu.edu.sg/home/ehchua/programming/java/javanativeinterface.html
#
#####################################################################

EXEFILE      = DemoC
LIBFILE      = libCellularAutomata-A-2D.so
CPUCC        = gcc
GPUCC        = /usr/local/cuda-12.2/bin/nvcc
CPPFLAGS     = -m64 #     -Wunused-variable #-Wno-conversion-null -Wdelete-non-virtual-dtor
DEFS        = $(APPLICATION)  $(VERSION) -DALIGN=64
INCLUDES    = -I. -I/usr/local/cuda/include -I/usr/lib/jvm/java-17-openjdk-amd64/include -I/usr/lib/jvm/java-17-openjdk-amd64/include/linux
LIBDIR      = -L. -L/usr/local/cuda/lib64
#LIBDIR      = -L/opt/gnu-gcc/lib64
LIBS     =  -lm  -lcuda -lcudart
LINK     =  $(LIBDIR) $(LIBS)
CPU_COMPILE = $(CPUCC) $(DEFS) $(INCLUDES) $(CPPFLAGS)
GPU_COMPILE = $(GPUCC) $(DEFS) $(INCLUDES) $(CPPFLAGS)

ifeq ($(target), release)
  DEFS      +=
  CPPFLAGS  += -O3
else
  DEFS     += -D_DEBUG
  CPPFLAGS += -g -O0
endif

all: exec

exec: kernel main
	$(CPU_COMPILE) obj/kernel.o obj/main.o $(LINK) -o $(EXEFILE)

main: 
	$(CPU_COMPILE) -c main.c -o obj/main.o

kernel: 
	$(GPU_COMPILE) -x=cu -c kernel.cu -o obj/kernel.o


clean:
	rm obj/*.o  $(EXEFILE) $(LIBFILE)

#####################################################################
#
# JNI Lib example
#
#####################################################################
JNI: jnikernel native buildjni

buildjni:
	$(CPU_COMPILE)  -shared  -Wl,-soname,$(LIBFILE) -o $(LIBFILE) obj/lkernel.o obj/NativeCellularAutomata.o $(LINK)

native:
	$(CPU_COMPILE) -fPIC -c NativeCellularAutomata.c -o obj/NativeCellularAutomata.o

jnikernel:
	$(GPU_COMPILE) -Xcompiler -fPIC -c kernel.cu -o obj/lkernel.o
