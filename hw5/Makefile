all: multi device pi test

multi:
	nvcc multipi.cu -o multipi.out

multi2:
	nvcc multipi2.cu -o multipi2.out

pi:
	nvcc calcpi.cu -o calcpi.out


clean:
	rm -f *.out
