#include<iostream>
#include<sstream>

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/ml/ml.hpp>

#include <opencv2/opencv.hpp>

const int MIN_CONTOUR_AREA = 70;

class ContourWithData {
public:
    std::vector<cv::Point> contour;
    cv::Rect contourRect;
    float fltArea;

    bool ifContourIsValid() 
    {
        if (fltArea < MIN_CONTOUR_AREA) 
            return false;
        else
            return true;                                 
    }

    static bool sortByBoundingRectXPosition(const ContourWithData& cwdLeft, const ContourWithData& cwdRight) 
    {
        if (cwdLeft.contourRect.x < cwdRight.contourRect.x)
            return true;
        else
            return false;
    }
};

int main(int argc, char** argv)
{
    cv::String path("images/*");
    std::vector<cv::String> fn;
    cv::glob(path, fn, true);

    cv::Mat matClassification;
    cv::Mat matImages;

    /*--------------------------------------------------------------------------*/

    cv::FileStorage fileCls("classifications.xml", cv::FileStorage::READ);
    if (fileCls.isOpened() == false)
    {
        std::cout << "Unable to open training classifications file\n";
        return(0);
    }
    fileCls["classifications"] >> matClassification;
    fileCls.release();
    std::cout << "OK. READ classifications: " << std::endl;


    cv::FileStorage fileImages("images.xml", cv::FileStorage::READ);
    if (fileImages.isOpened() == false)
    {
        std::cout << "Unable to open training images file\n";
        return(0);
    }
    fileImages["images"] >> matImages;
    fileImages.release();
    std::cout << "OK. READ IMAGES as mat: " << std::endl;

    /*--------------------------------------------------------------------------*/

    cv::Ptr<cv::ml::KNearest>  kNearest(cv::ml::KNearest::create());
    kNearest->train(matImages, cv::ml::ROW_SAMPLE, matClassification);

    /*--------------------------------------------------------------------------*/

    std::cout << "Number of images:  " << fn.size() << std::endl;
    if (fn.size() == 0) 
    {
        std::cout << "Could not find the image in path" << std::endl;
        return 0;
    }
  
    for (int k = 0; k < fn.size(); k++)
    {
        std::vector<ContourWithData> allContours;
        std::vector<ContourWithData> validContours;

        cv::Mat imageOrigin;
        cv::Mat imageGrayscale;
        cv::Mat imageBlurred;
        cv::Mat imageThresh;
        cv::Mat imageEroded;

        std::vector<std::vector<cv::Point> > contour;
        std::vector<cv::Vec4i> hierarchy;

        std::string strFinalString;

        cv::String windowName = fn[0];
        cv::namedWindow(windowName);

        imageOrigin = cv::imread(fn[k]);

        cv::cvtColor(imageOrigin, imageGrayscale, cv::COLOR_BGRA2GRAY);
        cv::imshow("imageGrayscale", imageGrayscale);
        cv::moveWindow("imageGrayscale", 50, 500);

        cv::GaussianBlur(imageGrayscale, imageBlurred, cv::Size(5, 5), 0);
        cv::imshow("imageBlurred", imageBlurred);
        cv::moveWindow("imageBlurred", 250, 500);

        cv::adaptiveThreshold(imageBlurred, imageThresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 11, 2);
        cv::imshow("imageThresh", imageThresh);
        cv::moveWindow("imageThresh", 500, 500);

        cv::erode(imageThresh, imageEroded, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2.5, 2.5)));
        cv::imshow("imageEroded", imageEroded);
        cv::moveWindow("imageEroded", 1000, 500);

        cv::findContours(imageEroded, contour, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (int i = 0; i < contour.size(); i++)
        {
            ContourWithData contourWithData;
            contourWithData.contour = contour[i];
            contourWithData.contourRect = cv::boundingRect(contourWithData.contour);
            contourWithData.fltArea = cv::contourArea(contourWithData.contour);
            allContours.push_back(contourWithData);
        }

        for (int i = 0; i < allContours.size(); i++)
            if (allContours[i].ifContourIsValid())
                validContours.push_back(allContours[i]);

        std::sort(validContours.begin(), validContours.end(), ContourWithData::sortByBoundingRectXPosition);

        for (int i = 0; i < validContours.size(); i++)
        {
            cv::Mat oneImg;
            cv::Mat oneImgResized;
            cv::Mat oneImgFloat;
            cv::Mat oneImgFlattenedFloat;
            cv::Mat matCurrentChar(0, 0, CV_32F);

            cv::rectangle(imageOrigin, validContours[i].contourRect, cv::Scalar(150, 100, 0), 2);
            oneImg = imageEroded(validContours[i].contourRect);

            cv::resize(oneImg, oneImgResized, cv::Size(30, 40));
            cv::imshow("one number", oneImg);
            cv::moveWindow("one number", 250, 250);
            cv::imshow(windowName, imageOrigin);
            cv::moveWindow(windowName, 1050, 50);

            oneImgResized.convertTo(oneImgFloat, CV_32FC1);
            oneImgFlattenedFloat = oneImgFloat.reshape(1, 1);

            kNearest->findNearest(oneImgFlattenedFloat, 1, matCurrentChar);
            float fltCurrentChar = (float)matCurrentChar.at<float>(0, 0);
            strFinalString = strFinalString + char(int(fltCurrentChar));
            cv::waitKey(0);
        }
        cv::destroyWindow(windowName);
        std::cout << std::endl;
        std::cout << "Result = " << strFinalString << std::endl;
    }

    cv::waitKey(0);                                    
    return(0);
}