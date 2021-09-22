#include<iostream>
#include<vector>

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/ml/ml.hpp>

const int MIN_CONTOUR_AREA = 100;

int main() {

    cv::Mat imageOrigin;
    cv::Mat imageGrayscale;
    cv::Mat imageBlurred;
    cv::Mat imageThresh;

    std::vector<std::vector<cv::Point> > contour;
    std::vector<cv::Vec4i> hierarchy;

    cv::Mat matClassification;
    cv::Mat matImages;

    std::vector<int> numbers = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'};

    /*--------------------------------------------------------------------------*/

    imageOrigin = cv::imread("test0.png");
    //cv::imshow("imageOrigin", imageOrigin);
    if (imageOrigin.empty()) {
        std::cout << "Image not found\n";
        return(0);
    }

    cv::cvtColor(imageOrigin, imageGrayscale, cv::COLOR_BGRA2GRAY);
    //cv::imshow("imageGrayscale", imageGrayscale);

    cv::GaussianBlur(imageGrayscale, imageBlurred, cv::Size(5, 5), 0);
    //cv::imshow("imageBlurred", imageBlurred);

    cv::adaptiveThreshold(imageBlurred, imageThresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 11, 2);
    //cv::imshow("imageThresh", imageThresh);

    cv::findContours(imageThresh, contour, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    /*--------------------------------------------------------------------------*/

    for (int i = 0; i < contour.size(); i++)
    {
        if (cv::contourArea(contour[i]) > MIN_CONTOUR_AREA)
        {
            cv::Rect contourRect;
            cv::Mat oneImg;
            cv::Mat oneImgResized;
            int pressedNumber;

            contourRect = cv::boundingRect(contour[i]);
            cv::rectangle(imageOrigin, contourRect, cv::Scalar(0, 0, 255), 2);
            oneImg = imageThresh(contourRect);

            cv::resize(oneImg, oneImgResized, cv::Size(30, 40));
            cv::imshow("one number", oneImg);
            cv::imshow("Image for training", imageOrigin);

            pressedNumber = cv::waitKey(0);
            auto tmp = std::find(numbers.begin(), numbers.end(), pressedNumber);
            if (tmp != numbers.end())
            {
                std::cout << "you pressed: " << pressedNumber << std::endl;
                matClassification.push_back(pressedNumber);

                cv::Mat matImageFloat;
                oneImgResized.convertTo(matImageFloat, CV_32FC1);
                cv::Mat matImageFlattenedFloat = matImageFloat.reshape(1, 1);
                matImages.push_back(matImageFlattenedFloat);
            }
        }
    }

    /*--------------------------------------------------------------------------*/

    cv::FileStorage fileCls("classifications.xml", cv::FileStorage::WRITE);
    if (fileCls.isOpened() == false)
    {
        std::cout << "Unable to open training classifications file\n";
        return(0);
    }
    fileCls << "classifications" << matClassification;
    fileCls.release();
    std::cout << "OK. WRITE classifications: " << matClassification << std::endl;


    cv::FileStorage fileImages("images.xml", cv::FileStorage::WRITE);
    if (fileImages.isOpened() == false)
    {
        std::cout << "Unable to open training images file\n";
        return(0);
    }
    fileImages << "images" << matImages;
    fileImages.release();
    std::cout << "OK. WRITE IMAGES as mat: " << matImages << std::endl;
}