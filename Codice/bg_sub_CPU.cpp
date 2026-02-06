#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

using namespace cv;
using namespace std;

// Parametri
#define ALPHA 0.01f
#define K 3.0f

void backgroundSubtractionCPU(
    const Mat& frame,      // grayscale uchar
    Mat& mean,             
    Mat& var,              
    Mat& mask              
) {
    int rows = frame.rows;
    int cols = frame.cols;
    int size = rows * cols;

    const uchar* f = frame.data;
    float* m = (float*)mean.data;
    float* v = (float*)var.data;
    uchar* out = mask.data;

    // kernel gaussiano 5x5 (normalizzato a 256)
    const int G[5][5] = {
        {1, 4, 6, 4, 1},
        {4,16,24,16,4},
        {6,24,36,24,6},
        {4,16,24,16,4},
        {1, 4, 6, 4, 1}
    };

    for (int idx = 0; idx < size; idx++) {

        int r = idx / cols;
        int c = idx % cols;

        float sum = 0.0f;

        //Gaussian blur 5x5
        for (int dr = -2; dr <= 2; dr++) {
            for (int dc = -2; dc <= 2; dc++) {
                int rr = min(max(r + dr, 0), rows - 1);
                int cc = min(max(c + dc, 0), cols - 1);
                sum += G[dr + 2][dc + 2] * f[rr * cols + cc];
            }
        }

        float pixel = sum / 256.0f;
        float diff  = pixel - m[idx];
        float stddev = sqrtf(v[idx] + 1e-6f);

        out[idx] = (fabs(diff) > K * stddev) ? 255 : 0;

        // update mean e var
        //con if aggiorni solamente se è sfondo
        if (fabs(diff) <= K * stddev) {
            m[idx] = (1.0f - ALPHA) * m[idx] + ALPHA * pixel;
            v[idx] = (1.0f - ALPHA) * v[idx] + ALPHA * diff * diff;
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Utilizzo: ./bg_sub video.mp4" << endl;
        return -1;
    }

    VideoCapture cap(argv[1]);
    if (!cap.isOpened()) return -1;

    Mat frame, grayFrame;
    Mat mean, var, mask;

    //inizializza primo frame
    cap >> frame;
    cvtColor(frame, grayFrame, COLOR_BGR2GRAY);

    mean = Mat(grayFrame.size(), CV_32FC1);
    var  = Mat(grayFrame.size(), CV_32FC1, Scalar(100.0f));
    mask = Mat::zeros(grayFrame.size(), CV_8UC1);

    //mean iniziale = primo frame
    grayFrame.convertTo(mean, CV_32FC1);

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cvtColor(frame, grayFrame, COLOR_BGR2GRAY);

        auto start = chrono::high_resolution_clock::now();

        backgroundSubtractionCPU(grayFrame, mean, var, mask);

        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> elapsed = end - start;

        cout << "Tempo elaborazione frame: "
             << elapsed.count() << " ms" << endl;

        imshow("Originale", frame);
        imshow("Maschera Movimento (CPU)", mask);

        if (waitKey(30) >= 0) break;
    }

    return 0;
}
