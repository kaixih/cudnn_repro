all:
	nvcc cudnn_v7_get_algorithms.cu -lcudnn
clean:
	rm -rf a.out
