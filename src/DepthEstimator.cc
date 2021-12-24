#include <vector>
#include <opencv2/imgproc.hpp>

#include "DepthEstimator.h"


namespace ORB_SLAM3
{
DepthEstimator::DepthEstimator(const std::string& encoderPath, const std::string& decoderPath, const int inputWidth, const int inputHeight, const std::string& device, const float scaleFactor):
    mInputSize{inputWidth, inputHeight}, mDevice{torch::kCPU}, mScaleFactor{scaleFactor}
{
    // Load encoder/decoder model
    mEncoder = torch::jit::load(encoderPath);
    mDecoder = torch::jit::load(decoderPath);

    // Set device
    if (device == "gpu")
        mDevice = torch::kCUDA;

    // Move models to device
    mEncoder.to(mDevice);
    mDecoder.to(mDevice);
}

cv::Mat DepthEstimator::EstimateDepth (const cv::Mat& imRGB)
{
    // Preprocess input
    cv::Mat inputImage;
    cv::resize(imRGB, inputImage, mInputSize);
    inputImage.convertTo(inputImage, CV_32FC3, 1. / 255.);
    torch::Tensor inputTensor{torch::from_blob(inputImage.data, {1, inputImage.rows, inputImage.cols, 3}, torch::kF32)};
    inputTensor = inputTensor.permute({0, 3, 1, 2});

    // Move input tensor to device
    inputTensor = inputTensor.to(mDevice);

    // Inference
    std::vector<torch::IValue> inputBatch;
    inputBatch.emplace_back(inputTensor);
    const auto outputEncoder{mEncoder.forward(inputBatch)};
    inputBatch.clear();
    inputBatch.emplace_back(outputEncoder);
    auto outputDecoder{mDecoder.forward(inputBatch)};

    // Move output tensor to cpu
    torch::Tensor outputTensor{outputDecoder.toTensor()};
    if (mDevice == torch::kCUDA)
        outputTensor = outputTensor.to(torch::kCPU);

    // Postprocess output
    outputTensor = outputTensor.permute({0, 3, 2, 1});
    cv::Mat disparity{mInputSize.height, mInputSize.width, CV_32FC1, outputTensor.data_ptr()};
    cv::resize(disparity, disparity, cv::Size(imRGB.cols, imRGB.rows));
    // Convert to metric depth (min_depth=0.01f, max_depth=10.0f)
    cv::Mat metricDepth{mScaleFactor / (0.01f + (10.0f - 0.01f) * disparity)};

    return metricDepth;
}

} //namespace ORB_SLAM
