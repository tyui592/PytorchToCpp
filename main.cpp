#include <iostream>
#include <string>
#include <typeinfo> // to check the type of variable
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include "argparse.hpp"

#define IMSIZE 512
#define MEAN {0.485, 0.456, 0.406}
#define STD {0.229, 0.224, 0.225}

bool LoadImage(std::string file_name, cv::Mat &image){
    image = cv::imread(file_name);
    if (image.empty() || !image.data){
        return false;
    }

    // color channel: bgr to rgb
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    std::cout << "image.size: " << image.size() << std::endl;

    // resize
    cv::Size scale(IMSIZE, IMSIZE);
    cv::resize(image, image, scale, 0, 0, cv::INTER_AREA);
    std::cout << "resize: " << image.size() << std::endl;

    // int[0~255] to float[0.0 ~ 1.0]
    image.convertTo(image, CV_32FC3, 1.0f/255.0f);

    return true;
}


int main(int argc, const char** argv){
    ArgumentParser parser;
    parser.addArgument("-m", "--model", 1);
    parser.addArgument("-i", "--image", 1);
    parser.parse(argc, argv);

    std::string model_path = parser.retrieve<std::string>("model");
    std::string image_path = parser.retrieve<std::string>("image");

    torch::data::transforms::Normalize<> normalize_transform(MEAN, STD); 
    torch::jit::script::Module model = torch::jit::load(model_path);
    model.to(at::kCUDA);
    model.eval();
    std::cout << "model load!" << std::endl;

    cv::Mat image;
    if (LoadImage(image_path, image)){
        // cv image to tensor
        at::Tensor tensor = torch::from_blob(image.data, {image.rows, image.cols, 3});
        tensor = tensor.permute({2, 0, 1});
        tensor = normalize_transform(tensor).unsqueeze_(0);
        tensor = tensor.to(torch::kCUDA);

        // forward
        // The first forwarding usually takes a long time, so forward twice to measure speed.
        auto output = model.forward({tensor}).toTuple()->elements();
        auto start = std::chrono::high_resolution_clock::now();
        output = model.forward({tensor}).toTuple()->elements();
        auto stop = std::chrono::high_resolution_clock::now();

        // measure speed
        std::chrono::duration<double> duration = stop - start;
        std::cout << "Inference Time: " << duration.count() * 1000 << "(ms)" << std::endl;

        // print prediction results
        std::cout << "Prediction Result(box, class, score)" << std::endl;
        auto box   = output[0].toTensor().to(torch::kCPU);
        auto cls   = output[1].toTensor().to(torch::kCPU);
        auto score = output[2].toTensor().to(torch::kCPU);

        // access tensor values 
        auto box_accessor   = box.accessor<float, 2>();
        auto cls_accessor   = cls.accessor<long, 1>();
        auto score_accessor = score.accessor<float, 1>();
        
        // color channel: rgb to bgr
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
        // float to uint8
        image.convertTo(image, CV_8UC3, 255);
        for (int i =0; i < box_accessor.size(0); i++){
            cv::Point pt1{};
            cv::Point pt2{};

            pt1.x = box_accessor[i][0];
            pt1.y = box_accessor[i][1];
            pt2.x = box_accessor[i][2];
            pt2.y = box_accessor[i][3];

            if (cls_accessor[i] == 0)
                cv::rectangle(image, pt1, pt2, cv::Scalar(0, 0, 255), 1);
            else
                cv::rectangle(image, pt1, pt2, cv::Scalar(0, 255, 0), 1);

            std::cout << "index: " << i
                      << ", xmin: " << pt1.x
                      << ", ymin: " << pt1.y
                      << ", xmax: " << pt2.x
                      << ", ymax: " << pt2.y
                      << ", class: " << cls_accessor[i]
                      << ", score: " << score_accessor[i]
                      << std::endl;
        }
        cv::imwrite("result.jpg", image);
        cv::imshow("image", image);
        cv::waitKey(3000);
        cv::destroyAllWindows();
    }
    return 0;
}
