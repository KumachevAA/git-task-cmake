#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main()
{
    ocl::setUseOpenCL(true);

    string image_path = "sample.jpg";
    Mat img = imread(image_path, IMREAD_COLOR);

    if (img.empty())
    {
        cout << "Could not read the image: " << image_path << endl;
        return 1;
    }

    Mat img_gray;
    cvtColor(img, img_gray, COLOR_BGR2GRAY);
    equalizeHist(img_gray, img_gray);

    string face_cascade_path = "haarcascade_frontalface_alt2.xml";
    string eyes_cascade_path = "haarcascade_eye.xml";

    CascadeClassifier face_cascade;
    CascadeClassifier eyes_cascade;

    if (!face_cascade.load(face_cascade_path))
    {
        cout << "Could not load face cascade: " << face_cascade_path << endl;
        return 1;
    }

    if (!eyes_cascade.load(eyes_cascade_path))
    {
        cout << "Could not load face cascade: " << eyes_cascade_path << endl;
        return 1;
    }

    vector<Rect> faces;
    face_cascade.detectMultiScale(img_gray, faces);
    for (auto& face: faces)
    {
        Mat face_gray = img_gray(face);

        vector<Rect> eyes;
        eyes_cascade.detectMultiScale(face_gray, eyes);

        for (auto& eye : eyes)
        {
            Rect eyePos = Rect(eye.x + face.x, eye.y + face.y, eye.width, eye.height);
            rectangle(img, eyePos, Scalar(0, 0, 0), 1);
        }

        rectangle(img, face, Scalar(0, 0, 0), 2);
    }

    imshow("Display window", img);
    waitKey(0);
    return 0;
}
