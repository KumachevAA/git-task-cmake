#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/ocl_genbase.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main()
{
    ocl::setUseOpenCL(true);
    VideoCapture video("video.mp4");

    if (!video.isOpened())
    {
        cout << "An error occured - could not load video" << endl;
        return 1;
    }

    CascadeClassifier faceFinder;

    if (!faceFinder.load("haarcascade_frontalface_alt.xml"))
    {
        cout << "An error occured - could not load cascade" << endl;
        return 1;
    }

    int ex = static_cast<int>(video.get(CAP_PROP_FOURCC));
    char EXT[] = { (char)(ex & 0XFF) , (char)((ex & 0XFF00) >> 8),(char)((ex & 0XFF0000) >> 16),(char)((ex & 0XFF000000) >> 24), 0 };
    cout << EXT << endl;

    float fps = video.get(CAP_PROP_FPS);
    int frames = (int) video.get(CAP_PROP_FRAME_COUNT);
    Size size = Size((int)video.get(CAP_PROP_FRAME_WIDTH), (int)video.get(CAP_PROP_FRAME_HEIGHT));

    VideoWriter writer("output.mp4", ex, fps, size, true);

    if (!writer.isOpened())
    {
        cout << "An error occured - could not open writer" << endl;
        return 1;
    }

    UMat frame;
    UMat frame_gray;
    int frame_num = 0;

    while (video.read(frame))
    {
        frame_num++;
        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
        equalizeHist(frame_gray, frame_gray);

        vector<Rect> faces;
        faceFinder.detectMultiScale(frame_gray, faces);

        for (auto& face: faces)
        {
            rectangle(frame, face, Scalar(0, 0, 255), 5);
        }

        writer.write(frame);

        cout << frame_num << " / " << frames << endl;
    }
}
