// onnxruntime_demo.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "pch.h"
#include <iostream>
#include <assert.h>
#include <vector>
#include <Windows.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

BOOL MByteToWChar(LPCSTR lpcszStr, LPWSTR lpwszStr, DWORD dwSize)
{
    // Get the required size of the buffer that receives the Unicode string.
    DWORD dwMinSize;
    dwMinSize = MultiByteToWideChar(CP_ACP, 0, lpcszStr, -1, NULL, 0);
    if (dwSize < dwMinSize)
        return FALSE;

    // Convert headers from ASCII to Unicode.
    MultiByteToWideChar(CP_ACP, 0, lpcszStr, -1, lpwszStr, dwMinSize);
    return TRUE;
}

int main(int argc, char* argv[]) {
    if (argc < 3)
    {
        cout << "usage : onnxruntime_demo.exe <picture path> <onnx model path>  <NumThreads:default 1>" << endl;
        return 0;
    }

    char * pPicturePath = argv[1];
    char * pModelPath = argv[2];
    int    nNumThreads = 1;

    if (argc == 4) {
        nNumThreads = atoi(argv[3]);
    }
    //char * pDeviceType = argv[4];
    WCHAR  wszModelPath[1000];
    MByteToWChar(pModelPath, wszModelPath, 256);
  //*************************************************************************
  // initialize  enviroment...one enviroment per process
  // enviroment maintains thread pools and other state info
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

  // initialize session options if needed
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(nNumThreads);

  // Sets graph optimization level
  // Available levels are
  // ORT_DISABLE_ALL -> To disable all optimizations
  // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node removals)
  // ORT_ENABLE_EXTENDED -> To enable extended optimizations (Includes level 1 + more complex optimizations like node fusions)
  // ORT_ENABLE_ALL -> To Enable All possible opitmizations
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  //*************************************************************************
  printf("Using Onnxruntime C++ API\n");
  Ort::Session session(env, wszModelPath, session_options);

  //*************************************************************************
  // print model input layer (node names, types, shape etc.)
  Ort::AllocatorWithDefaultOptions allocator;

  // print number of model input nodes
  size_t num_input_nodes = session.GetInputCount();
  std::vector<const char*> input_node_names(num_input_nodes);
  std::vector<int64_t> input_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
                                         // Otherwise need vector<vector<>>

  printf("Number of inputs = %zu\n", num_input_nodes);

  // iterate over all input nodes
  for (int i = 0; i < num_input_nodes; i++) {
    // print input node names
    char* input_name = session.GetInputName(i, allocator);
    printf("Input %d : name=%s\n", i, input_name);
    input_node_names[i] = input_name;

    // print input node types
    Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType type = tensor_info.GetElementType();
    printf("Input %d : type=%d\n", i, type);

    // print input shapes/dims
    input_node_dims = tensor_info.GetShape();
    printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
    for (int j = 0; j < input_node_dims.size(); j++)
      printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
  }


  //*************************************************************************
  // Similar operations to get output node information.
  // print number of model output nodes
  size_t num_output_nodes = session.GetOutputCount();
  std::vector<const char*> output_node_names(num_output_nodes);
  std::vector<int64_t> output_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
  
  printf("Number of outputs = %zu\n", num_output_nodes);
  // iterate over all input nodes
  for (int i = 0; i < num_output_nodes; i++) {
      // print input node names
      char* output_name = session.GetOutputName(i, allocator);
      printf("Input %d : name=%s\n", i, output_name);
      output_node_names[i] = output_name;

      // print input node types
      Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

      ONNXTensorElementDataType type = tensor_info.GetElementType();
      printf("Output %d : type=%d\n", i, type);

      // print input shapes/dims
      output_node_dims = tensor_info.GetShape();
      printf("Output %d : num_dims=%zu\n", i, output_node_dims.size());
      for (int j = 0; j < output_node_dims.size(); j++)
          printf("Output %d : dim %d=%jd\n", i, j, output_node_dims[j]);
  }

  //*************************************************************************
  // Score the model using sample data, and inspect values
  size_t input_tensor_size = input_node_dims[0] * input_node_dims[1] * input_node_dims[2] * input_node_dims[3];
  //size_t input_tensor_size = 224 * 224 * 3;  // simplify ... using known dim values to calculate size

  std::vector<float> input_tensor_values(input_tensor_size);
  std::vector<const char*> output_node_names_real = {"Output"};

  // initialize input data with values in [0.0, 1.0]
  
  Mat matdata = imread(pPicturePath, IMREAD_COLOR);
  Mat resizedData(input_node_dims[2], input_node_dims[3], CV_8UC3);
  resize(matdata, resizedData, resizedData.size());

  size_t num_channels = input_node_dims[1];
  size_t image_size = input_node_dims[2]*input_node_dims[3];

  /** Iterate over all pixel in image (r,g,b) **/
  for (size_t pid = 0; pid < image_size; pid++) {
      /** Iterate over all channels **/
      for (size_t ch = 0; ch < num_channels; ++ch) {
          input_tensor_values[ch * image_size + pid] = resizedData.data[pid*num_channels + ch];
      }
  }

  // create input tensor object from data values
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
  assert(input_tensor.IsTensor());

  // score model & input tensor, get back output tensor
  //auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names_real.data(), 1);
  
  int nTimes = 100;
  DWORD ticks = GetTickCount(), ticks2;
  for (size_t i = 0; i < nTimes; i++)
  {
      auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), &input_tensor, 1, output_node_names_real.data(), 1);
  }
  ticks2 = GetTickCount();
  DWORD  totalticks = (ticks2 - ticks);
  printf("WinVideoPortrait::Process after predict,total =%d ms(%d times), average=%.2f ms\n", totalticks, nTimes, (float)totalticks / nTimes);


  auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), &input_tensor, 1, output_node_names_real.data(), 1);
  assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

  // Get pointer to output tensor float values
  float* floatarr = output_tensors.front().GetTensorMutableData<float>();
  size_t C = 1, H = 224, W = 224;
  Mat outputMaskData(224, 224, CV_8UC1);
  for (size_t h = 0; h < H; ++h) {
    for (size_t w = 0; w < W; ++w) {
          outputMaskData.data[W * h + w] = static_cast<uint8_t>(floatarr[W * h + w] * 255);
          if (outputMaskData.data[W * h + w] < 128) {
              resizedData.data[(W * h + w)*num_channels] = 255;
              resizedData.data[(W * h + w)*num_channels + 1] = 255;
              resizedData.data[(W * h + w)*num_channels + 2] = 255;
          }
      }
  }

  imwrite("output_mask.bmp", outputMaskData);
  imwrite("output_seg.bmp", resizedData);

  printf("File:output_mask.bmp and output_seg.bmp were created, Done!\n");
  return 0;
}