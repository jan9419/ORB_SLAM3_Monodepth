#pragma once

#include <string>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <torch/script.h>
#include <torch/torch.h>


namespace ORB_SLAM3
{

// Class for monocular depth estimation based on a deep neural network
class DepthEstimator
{
public:

    DepthEstimator(const std::string& encoderPath, const std::string& decoderPath, const int inputWidth, const int inputHeight, const std::string& device, const float scaleFactor);

    // Estimate metric depth with deep neural network
    // Input: RGB image
    cv::Mat EstimateDepth(const cv::Mat& imRGB);

private:

    // Encoder model
    torch::jit::script::Module mEncoder;

    // Decoder model
    torch::jit::script::Module mDecoder;

    // Enocder input size
    cv::Size mInputSize;

    // Device (cpu, gpu)
    torch::Device mDevice;

    // Stereo scale factor
    float mScaleFactor;
};

} //namespace ORB_SLAM
