#include <iostream>
#include <cudnn.h>
#include <cuda_fp16.h>

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

int main(int argc, char const *argv[]) {
  cudnnHandle_t cudnn;
  checkCUDNN(cudnnCreate(&cudnn));

  int N = 1, C = 32, Dx = 320, Hx = 320, Wx = 320;
  int Dy = 160, Hy = 160, Wy = 160;
  int c = 32, k = 32, t = 2, r = 2, s = 2;
  int dx_dims[] = {N, C, Dx, Hx, Wx};
  int dy_dims[] = {N, C, Dy, Hy, Wy};
  int w_dims[] = {k, c, t, r, s};
  int paddings[] = {0, 0, 0};
  int strides[] = {2, 2, 2};
  int dilations[] = {1, 1, 1};

  /* Below is to do BackwardData */
  cudnnTensorDescriptor_t dy_desc;
  checkCUDNN(cudnnCreateTensorDescriptor(&dy_desc));
  checkCUDNN(cudnnSetTensorNdDescriptorEx(/*tensorDesc=*/dy_desc,
                                          /*format=*/CUDNN_TENSOR_NHWC,
                                          /*dataType=*/CUDNN_DATA_HALF,
                                          /*nbDims=*/5,
                                          /*dimsA*/dy_dims));

  cudnnTensorDescriptor_t dx_desc;
  checkCUDNN(cudnnCreateTensorDescriptor(&dx_desc));
  checkCUDNN(cudnnSetTensorNdDescriptorEx(/*tensorDesc=*/dx_desc,
                                          /*format=*/CUDNN_TENSOR_NHWC,
                                          /*dataType=*/CUDNN_DATA_HALF,
                                          /*nbDims=*/5,
                                          /*dimsA*/dx_dims));

  cudnnFilterDescriptor_t w_desc;
  checkCUDNN(cudnnCreateFilterDescriptor(&w_desc));
  checkCUDNN(cudnnSetFilterNdDescriptor(/*filterDesc=*/w_desc,
                                        /*dataType=*/CUDNN_DATA_HALF,
                                        /*format=*/CUDNN_TENSOR_NHWC,
                                        /*nbDims=*/5,
                                        /*dimsA*/w_dims));

  cudnnConvolutionDescriptor_t conv_desc;
  checkCUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
  checkCUDNN(cudnnSetConvolutionNdDescriptor(
      /*conv_desc=*/conv_desc,
      /*arrayLength=*/3,
      /*padA=*/paddings,
      /*filterStrideA=*/strides,
      /*dilationA=*/dilations,
      /*mode=*/CUDNN_CROSS_CORRELATION,
      /*dataType=*/CUDNN_DATA_FLOAT));

  // checkCUDNN(cudnnSetConvolutionMathType(conv_desc, CUDNN_FMA_MATH));
  checkCUDNN(cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH));

  int algos[5] = {1, 2, 3, 4, 0};
  for(auto algo : algos) {
    size_t size_in_bytes = 0;
    cudnnStatus_t status = cudnnGetConvolutionBackwardDataWorkspaceSize(
        /*handle=*/cudnn,
        /*wDesc=*/w_desc,
        /*dyDesc=*/dy_desc,
        /*convDesc=*/conv_desc,
        /*dxDesc=*/dx_desc,
        /*algo=*/(cudnnConvolutionBwdDataAlgo_t)algo,
        /*sizeInBytes=*/&size_in_bytes);
    if (status != CUDNN_STATUS_SUCCESS) {
      printf("XXX algo %d failed\n", algo);
    } else {
      printf("XXX found algo %d workspace size in bytes: %ld\n", algo, size_in_bytes);
    }
  }
}
