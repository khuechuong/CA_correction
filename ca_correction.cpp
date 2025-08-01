#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

void rmCA_(std::vector<cv::Mat> &bgrVec, int threshold)
{
    int height = bgrVec[0].rows, width = bgrVec[0].cols;

    cv::parallel_for_(cv::Range(0, height), [&](const cv::Range& range) {
        for (int i = range.start; i < range.end; ++i)
        {
            uchar *bptr = bgrVec[0].ptr<uchar>(i);
            uchar *gptr = bgrVec[1].ptr<uchar>(i);
            uchar *rptr = bgrVec[2].ptr<uchar>(i);

            for (int j = 1; j < width - 1; ++j)
            {
                if (std::abs(gptr[j + 1] - gptr[j - 1]) >= threshold)
                {
                    int sign = (gptr[j + 1] - gptr[j - 1] > 0) ? 1 : -1;

                    int lpos = j - 1, rpos = j + 1;
                    for (; lpos > 0; --lpos)
                    {
                        int ggrad = (gptr[lpos + 1] - gptr[lpos - 1]) * sign;
                        int bgrad = (bptr[lpos + 1] - bptr[lpos - 1]) * sign;
                        int rgrad = (rptr[lpos + 1] - rptr[lpos - 1]) * sign;
                        if (std::max({ bgrad, ggrad, rgrad }) < threshold) break;
                    }
                    lpos -= 1;

                    for (; rpos < width - 1; ++rpos)
                    {
                        int ggrad = (gptr[rpos + 1] - gptr[rpos - 1]) * sign;
                        int bgrad = (bptr[rpos + 1] - bptr[rpos - 1]) * sign;
                        int rgrad = (rptr[rpos + 1] - rptr[rpos - 1]) * sign;
                        if (std::max({ bgrad, ggrad, rgrad }) < threshold) break;
                    }
                    rpos += 1;

                    int bgmaxVal = std::max(bptr[lpos] - gptr[lpos], bptr[rpos] - gptr[rpos]);
                    int bgminVal = std::min(bptr[lpos] - gptr[lpos], bptr[rpos] - gptr[rpos]);
                    int rgmaxVal = std::max(rptr[lpos] - gptr[lpos], rptr[rpos] - gptr[rpos]);
                    int rgminVal = std::min(rptr[lpos] - gptr[lpos], rptr[rpos] - gptr[rpos]);

                    for (int k = lpos; k <= rpos; ++k)
                    {
                        int bdiff = bptr[k] - gptr[k];
                        int rdiff = rptr[k] - gptr[k];

                        bptr[k] = cv::saturate_cast<uchar>(
                            bdiff > bgmaxVal ? bgmaxVal + gptr[k] :
                            (bdiff < bgminVal ? bgminVal + gptr[k] : bptr[k])
                        );

                        rptr[k] = cv::saturate_cast<uchar>(
                            rdiff > rgmaxVal ? rgmaxVal + gptr[k] :
                            (rdiff < rgminVal ? rgminVal + gptr[k] : rptr[k])
                        );
                    }
                    j = rpos - 2;
                }
            }
        }
    });
}
void parallelSplit(const cv::Mat &Src, std::vector<cv::Mat> &bgrVec) {
    bgrVec.resize(3);
    for (int i = 0; i < 3; ++i)
        bgrVec[i] = cv::Mat(Src.rows, Src.cols, CV_8UC1);

    cv::parallel_for_(cv::Range(0, Src.rows), [&](const cv::Range &range) {
        for (int i = range.start; i < range.end; ++i) {
            const cv::Vec3b *srcRow = Src.ptr<cv::Vec3b>(i);
            uchar *bRow = bgrVec[0].ptr<uchar>(i);
            uchar *gRow = bgrVec[1].ptr<uchar>(i);
            uchar *rRow = bgrVec[2].ptr<uchar>(i);
            for (int j = 0; j < Src.cols; ++j) {
                bRow[j] = srcRow[j][0];
                gRow[j] = srcRow[j][1];
                rRow[j] = srcRow[j][2];
            }
        }
    });
}

void CACorrection(const cv::Mat &Src, cv::Mat &Dst)
{    
    std::vector<cv::Mat> bgrVec(3);
    // cv::split(Src, bgrVec);
    parallelSplit(Src, bgrVec);

    int threshold = 10;
    rmCA_(bgrVec, threshold);

    cv::parallel_for_(cv::Range(0, 3), [&](const cv::Range& range) {
        for (int i = range.start; i < range.end; ++i) {
            bgrVec[i] = bgrVec[i].t();
        }
    });

    rmCA_(bgrVec, threshold);

    // cv::merge(bgrVec, Dst);
    Dst.create(bgrVec[0].size(), CV_8UC3);
    cv::parallel_for_(cv::Range(0, Dst.rows), [&](const cv::Range &range) {
        for (int i = range.start; i < range.end; ++i) {
            cv::Vec3b* dst_ptr = Dst.ptr<cv::Vec3b>(i);
            const uchar* b_ptr = bgrVec[0].ptr<uchar>(i);
            const uchar* g_ptr = bgrVec[1].ptr<uchar>(i);
            const uchar* r_ptr = bgrVec[2].ptr<uchar>(i);
            for (int j = 0; j < Dst.cols; ++j) {
                dst_ptr[j][0] = b_ptr[j];
                dst_ptr[j][1] = g_ptr[j];
                dst_ptr[j][2] = r_ptr[j];
            }
        }
    });
    // Dst = Dst.t();
    cv::Mat tmp = Dst;  // Shallow copy; avoids in-place mutation risk
    cv::transpose(tmp, Dst);
}

void CACorrection_time(const cv::Mat &Src, cv::Mat &Dst)
{    
    std::vector<cv::Mat> bgrVec(3);
    auto t1 = std::chrono::high_resolution_clock::now();
    // cv::split(Src, bgrVec);
    parallelSplit(Src, bgrVec);

    auto t2 = std::chrono::high_resolution_clock::now();
    int threshold = 10;

    rmCA_(bgrVec, threshold);

    auto t3 = std::chrono::high_resolution_clock::now();
    // for (auto &channel : bgrVec)
    //     channel = channel.t();
    cv::parallel_for_(cv::Range(0, 3), [&](const cv::Range& range) {
        for (int i = range.start; i < range.end; ++i) {
            bgrVec[i] = bgrVec[i].t();
        }
    });

    auto t4 = std::chrono::high_resolution_clock::now();
    rmCA_(bgrVec, threshold);

    auto t5 = std::chrono::high_resolution_clock::now();

    // cv::merge(bgrVec, Dst);
    Dst.create(bgrVec[0].size(), CV_8UC3);
    cv::parallel_for_(cv::Range(0, Dst.rows), [&](const cv::Range &range) {
        for (int i = range.start; i < range.end; ++i) {
            cv::Vec3b* dst_ptr = Dst.ptr<cv::Vec3b>(i);
            const uchar* b_ptr = bgrVec[0].ptr<uchar>(i);
            const uchar* g_ptr = bgrVec[1].ptr<uchar>(i);
            const uchar* r_ptr = bgrVec[2].ptr<uchar>(i);
            for (int j = 0; j < Dst.cols; ++j) {
                dst_ptr[j][0] = b_ptr[j];
                dst_ptr[j][1] = g_ptr[j];
                dst_ptr[j][2] = r_ptr[j];
            }
        }
    });
    auto t6 = std::chrono::high_resolution_clock::now();
    // Dst = Dst.t();
    cv::Mat tmp = Dst;  // Shallow copy; avoids in-place mutation risk
    cv::transpose(tmp, Dst);
    auto t7 = std::chrono::high_resolution_clock::now();
    
    double elapsed_ms0 = std::chrono::duration<double, std::milli>(t2 - t1).count();
    double elapsed_ms1 = std::chrono::duration<double, std::milli>(t3 - t2).count();
    double elapsed_ms2 = std::chrono::duration<double, std::milli>(t4 - t3).count();
    double elapsed_ms3 = std::chrono::duration<double, std::milli>(t5 - t4).count();
    double elapsed_ms4 = std::chrono::duration<double, std::milli>(t6 - t5).count();
    double elapsed_ms5 = std::chrono::duration<double, std::milli>(t7 - t6).count();

    std::cout << "Split time: " << elapsed_ms0 << " ms" << std::endl;
    std::cout << "rmCA time: " << elapsed_ms1 << " ms" << std::endl;
    std::cout << "3 .t time: " << elapsed_ms2 << " ms" << std::endl;
    std::cout << "rmCA time: " << elapsed_ms3 << " ms" << std::endl;
    std::cout << "merge time: " << elapsed_ms4 << " ms" << std::endl;
    std::cout << ".t time: " << elapsed_ms5 << " ms" << std::endl;
}


int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cerr << "Usage: ./ca_correction <input_image> <output_image>" << std::endl;
        return 1;
    }

    cv::Mat img = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (img.empty())
    {
        std::cerr << "❌ Failed to load image: " << argv[1] << std::endl;
        return 1;
    }
    // Start timer
    auto t_start = std::chrono::high_resolution_clock::now();
    cv::Mat corrected;
    CACorrection_time(img, corrected);
    // Stop timer
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    std::cout << "Processing time: " << elapsed_ms << " ms" << std::endl;
    cv::imwrite(argv[2], corrected);
    std::cout << "✅ Saved corrected image to " << argv[2] << std::endl;
    return 0;
}