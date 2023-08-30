// Purpose of this file is to run a series of tests on several images using different implementations of the
// Recursive Bilaterial Filter, and to show rough time estimate for each run

#include "stdafx.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include "rbf.hpp"
#include <time.h>
#include <iostream>
#include <time.h>
#include "RBFilter_SSE2.h"
#include "RBFilter_AVX2.h"
#include <iomanip>
//#include <stdlib.h>
#include <malloc.h>

using namespace std;

// main filter strength controls
const float sigma_spatial = 0.12f;
const float sigma_range = 0.09f;

// number of test runs per image, for better average time measurement
// if running debug mode, use small number so it's faster
#ifdef _DEBUG
const int test_runs = 1;  
#else
const int test_runs = 100;
#endif

// path where files are located, you may need to change this
const char images_folder_path[] = "./images/";

// test images:
const char file_name_testGirl[] = "testGirl.jpg";		// size: 448 x 626
const char file_name_house[] = "Thefarmhouse.jpg";		// size: 1440 x 1080
const char file_name_testpattern[] = "testpatern5.png"; // size: 1920 x 1080


// timer uses 'test_runs' as divisor
class TestRunTimer 
{
	clock_t begTime;

public:
	void start() { begTime = clock(); }
	float elapsedTimeMS() { return (float(clock() - begTime) / (float)test_runs * 1000) / CLOCKS_PER_SEC; }
};

// utility for setting output file name
template <size_t _Size>
char* modifyFilePath(char (&file_path)[_Size], const char* suffix)
{
	size_t l = strlen(file_path);
	// get rid of old extension
	for (size_t i = l - 1; i > 0; i--)
	{
		if (file_path[i] == '.')
		{
			file_path[i] = 0;
			break;
		}
	}

	// add current sigma values just for clarity
	char extra_text[64];
	sprintf(extra_text, "%0.3f_%0.3f", sigma_spatial, sigma_range);

	// add suffix
	strcat(file_path, "_");
	strcat(file_path, suffix);
	strcat(file_path, "_");
	strcat(file_path, extra_text);
	strcat(file_path, ".png"); // force PNG format

	return file_path;
}

// using original implementation, source code from
// https://github.com/ufoym/RecursiveBF
void testRunRecursiveBF_Original(const char* image_name)
{
	cout << "\nImage: " << image_name;
	char file_path[256];
	strcpy(file_path, images_folder_path);
	strcat(file_path, image_name);

	int width, height, channel;
	unsigned char * img = stbi_load(file_path, &width, &height, &channel, 3);
	if (!img)
	{
		cout << "\nFailed to load image path: " << file_path;
		return;
	}
	cout << ", size: " << width << " x " << height;
	channel = 3; // require 3 channel for this test
	unsigned char * img_out = nullptr;
	TestRunTimer timer;

	// memory reserve for filter algorithm before timer start
	float * buffer = new float[(width * height* channel + width * height + width * channel + width) * 2];

	timer.start();
	for (int i = 0; i < test_runs; ++i)
		recursive_bf(img, img_out, sigma_spatial, sigma_range, width, height, channel, buffer);
	
	cout << ", time ms: " << timer.elapsedTimeMS();
	
	delete[] buffer;

	modifyFilePath(file_path, "RBF");
	stbi_write_png(file_path, width, height, channel, img_out, width * 3);

	delete[] img;
	delete[] img_out;
}


// using optimized SSE2 with optional multithreading, single stage (non-pipelined)
void testRunRecursiveBF_SSE2_mt(const char* image_name, int thread_count)
{
	cout << "\nImage: " << image_name;
	char file_path[256];
	strcpy(file_path, images_folder_path);
	strcat(file_path, image_name);

	int width, height, channel;
	unsigned char * img = stbi_load(file_path, &width, &height, &channel, 4);
	if (!img)
	{
		cout << "\nFailed to load image path: " << file_path;
		return;
	}
	cout << ", size: " << width << " x " << height;
	channel = 4; // require 4 channel for this test

	CRBFilterSSE2 rbf_object;
	bool success = rbf_object.initialize(width, height, thread_count, false);
	if (!success)
	{
		cout << "\nCRBFilterSSE2 failed to initialize for some reason";
		delete[] img;
		return;
	}
	rbf_object.setSigma(sigma_spatial, sigma_range);

	unsigned char * img_out = new unsigned char[width * height * 4];


	TestRunTimer timer;
	timer.start();
	
	for (int i = 0; i < test_runs; ++i)
		success = rbf_object.filter(img_out, img, width, height, width * 4);

	if (success)
	{
		cout << ", time ms: " << timer.elapsedTimeMS();
	}
	else // fail
	{
		cout << "\nCRBFilterSSE2::filter failed for some reason";
	}

	char suffix[64];
	sprintf(suffix, "SSE2_%dt", thread_count);
	modifyFilePath(file_path, suffix);
	stbi_write_png(file_path, width, height, channel, img_out, width * 4);

	delete[] img;
	delete[] img_out;
}

// using optimized SSE2 with optional multithreading, pipelined 2 stages
void testRunRecursiveBF_SSE2_Pipelined(const char* image_name, int thread_count)
{
	cout << "\nImage: " << image_name;
	char file_path[256];
	strcpy(file_path, images_folder_path);
	strcat(file_path, image_name);

	int width, height, channel;
	unsigned char * img = stbi_load(file_path, &width, &height, &channel, 4);
	if (!img)
	{
		cout << "\nFailed to load image path: " << file_path;
		return;
	}
	cout << ", size: " << width << " x " << height;
	channel = 4; // require 4 channel for this test

	CRBFilterSSE2 rbf_object;
	bool success = rbf_object.initialize(width, height, thread_count, true);
	if (!success)
	{
		cout << "\nCRBFilterSSE2 failed to initialize for some reason";
		delete[] img;
		return;
	}
	rbf_object.setSigma(sigma_spatial, sigma_range);

	// need 2 output buffers, one for each stage
	unsigned char * img_out[2];
	img_out[0] = new unsigned char[width * height * 4];
	img_out[1] = new unsigned char[width * height * 4];

	TestRunTimer timer;
	timer.start();

	for (int i = 0; i < test_runs; ++i)
		success = rbf_object.filterPipePush(img_out[i&1], img, width, height, width * 4);
	
	rbf_object.filterPipeFlush();

	if (success)
	{
		cout << ", time ms: " << timer.elapsedTimeMS();
	}
	else // fail
	{
		cout << "\nCRBFilterSSE2::filterPipePush failed for some reason";
	}

	char suffix[64];
	sprintf(suffix, "SSE2_Pipe_%dt", thread_count);
	modifyFilePath(file_path, suffix);
	stbi_write_png(file_path, width, height, channel, img_out[0], width * 4);

	delete[] img;
	delete[] img_out[0];
	delete[] img_out[1];
}


// using optimized AVX2 with optional multithreading, single stage (non-pipelined)
void testRunRecursiveBF_AVX2_mt(const char* image_name, int thread_count)
{
	cout << "\nImage: " << image_name;
	char file_path[256];
	strcpy(file_path, images_folder_path);
	strcat(file_path, image_name);

	int width, height, channel;
	unsigned char * img = stbi_load(file_path, &width, &height, &channel, 4);
	if (!img)
	{
		cout << "\nFailed to load image path: " << file_path;
		return;
	}
	cout << ", size: " << width << " x " << height;
	channel = 4; // require 4 channel for this test

	CRBFilterAVX2 rbf_object;
	bool success = rbf_object.initialize(width, height, thread_count, false);
	if (!success)
	{
		cout << "\nCRBFilterAVX2 failed to initialize for some reason";
		delete[] img;
		return;
	}
	rbf_object.setSigma(sigma_spatial, sigma_range);

	int pitch = rbf_object.getOptimalPitch(width);
	unsigned char * img_out;

	// setup 32 byte aligned memory buffers for input and output, using optimal pitch
	{
//		img_out = (unsigned char*)_aligned_malloc(pitch * height, 32);
		img_out = (unsigned char*)memalign(32, pitch * height);

		// move source image to aligned memory
//		unsigned char* buffer = (unsigned char*)_aligned_malloc(pitch * height, 32);
		unsigned char* buffer = (unsigned char*)memalign(32, pitch * height);
		for (int y = 0; y < height; y++)
		{
			memcpy(buffer + y * pitch, img + y * width * 4, width * 4);
		}
		delete[] img;
		img = buffer;
	}

	TestRunTimer timer;
	timer.start();

	for (int i = 0; i < test_runs; ++i)
		success = rbf_object.filter(img_out, img, width, height, pitch);

	if (success)
	{
		cout << ", time ms: " << timer.elapsedTimeMS();
	}
	else // fail
	{
		cout << "\nCRBFilterAVX2::filter failed for some reason";
	}

	char suffix[64];
	sprintf(suffix, "AVX2_%dt", thread_count);
	modifyFilePath(file_path, suffix);
	stbi_write_png(file_path, width, height, channel, img_out, pitch);

	free(img);
	free(img_out);
}

// using optimized AVX2 with optional multithreading, pipelined 2 stages, memory aligned
void testRunRecursiveBF_AVX2_Pipelined(const char* image_name, int thread_count)
{
	cout << "\nImage: " << image_name;
	char file_path[256];
	strcpy(file_path, images_folder_path);
	strcat(file_path, image_name);

	int width, height, channel;
	unsigned char * img = stbi_load(file_path, &width, &height, &channel, 4);
	if (!img)
	{
		cout << "\nFailed to load image path: " << file_path;
		return;
	}
	cout << ", size: " << width << " x " << height;
	channel = 4; // require 4 channel for this test

	CRBFilterAVX2 rbf_object;
	bool success = rbf_object.initialize(width, height, thread_count, true);
	if (!success)
	{
		cout << "\nCRBFilterAVX2 failed to initialize for some reason";
		delete[] img;
		return;
	}
	rbf_object.setSigma(sigma_spatial, sigma_range);

	int pitch = rbf_object.getOptimalPitch(width);
	unsigned char* img_out[2];

	// setup 32 byte aligned memory buffers for input and output, using optimal pitch
	{
//		img_out[0] = (unsigned char*)_aligned_malloc(pitch * height, 32);
//		img_out[1] = (unsigned char*)_aligned_malloc(pitch * height, 32);
		img_out[0] = (unsigned char*)memalign(32, pitch * height);
		img_out[1] = (unsigned char*)memalign(32, pitch * height);

		// move source image to aligned memory
//		unsigned char* buffer = (unsigned char*)_aligned_malloc(pitch * height, 32);
		unsigned char* buffer = (unsigned char*)memalign(32, pitch * height);
		for (int y = 0; y < height; y++)
		{
			memcpy(buffer + y * pitch, img + y * width * 4, width * 4);
		}
		delete[] img;
		img = buffer;
	}

	TestRunTimer timer;
	timer.start();

	for (int i = 0; i < test_runs; ++i)
		success = rbf_object.filterPipePush(img_out[i & 1], img, width, height, width * 4);

	rbf_object.filterPipeFlush();

	if (success)
	{
		cout << ", time ms: " << timer.elapsedTimeMS();
	}
	else // fail
	{
		cout << "\nCRBFilterAVX2::filterPipePush failed for some reason";
	}

	char suffix[64];
	sprintf(suffix, "AVX2_Pipe_%dt", thread_count);
	modifyFilePath(file_path, suffix);
	stbi_write_png(file_path, width, height, channel, img_out[0], pitch);

	free(img);
	free(img_out[0]);
	free(img_out[1]);
}

/////////////////////////////////////////////////////////////////////////////

int main()
{
	cout << "test run \n";
	cout << fixed << setprecision(1);

	////////////////////////
	cout << "\nOriginal Recursive Bilateral Filter implementation";
	// image: testpattern
	testRunRecursiveBF_Original(file_name_testpattern);
	// image: house
	testRunRecursiveBF_Original(file_name_house);
	// image: testGirl
	testRunRecursiveBF_Original(file_name_testGirl);

	
	////////////////////////
	cout << "\n\nOptimized SSE2 single threaded, single stage (non-pipelined)";
	// image: testpattern
	testRunRecursiveBF_SSE2_mt(file_name_testpattern, 1);
	// image: house
	testRunRecursiveBF_SSE2_mt(file_name_house, 1);
	// image: testGirl
	testRunRecursiveBF_SSE2_mt(file_name_testGirl, 1);

	////////////////////////
	cout << "\n\nOptimized SSE2 2x multithreading, single stage (non-pipelined)";
	// image: testpattern
	testRunRecursiveBF_SSE2_mt(file_name_testpattern, 2);
	// image: house
	testRunRecursiveBF_SSE2_mt(file_name_house, 2);
	// image: testGirl
	testRunRecursiveBF_SSE2_mt(file_name_testGirl, 2);

	////////////////////////
	cout << "\n\nOptimized SSE2 4x multithreading, single stage (non-pipelined)";
	// image: testpattern
	testRunRecursiveBF_SSE2_mt(file_name_testpattern, 4);
	// image: house
	testRunRecursiveBF_SSE2_mt(file_name_house, 4);
	// image: testGirl
	testRunRecursiveBF_SSE2_mt(file_name_testGirl, 4);

	////////////////////////
	cout << "\n\nOptimized SSE2 4x2 thread pipelined 2 stages";
	// image: testpattern
	testRunRecursiveBF_SSE2_Pipelined(file_name_testpattern, 4);
	// image: house
	testRunRecursiveBF_SSE2_Pipelined(file_name_house, 4);
	// image: testGirl
	testRunRecursiveBF_SSE2_Pipelined(file_name_testGirl, 4);

	////////////////////////
	cout << "\n\nOptimized AVX2 single threaded, single stage (non-pipelined), memory aligned";
	// image: testpattern
	testRunRecursiveBF_AVX2_mt(file_name_testpattern, 1);
	// image: house
	testRunRecursiveBF_AVX2_mt(file_name_house, 1);
	// image: testGirl
	testRunRecursiveBF_AVX2_mt(file_name_testGirl, 1);

	////////////////////////
	cout << "\n\nOptimized AVX2 2x multithreading, single stage (non-pipelined), memory aligned";
	// image: testpattern
	testRunRecursiveBF_AVX2_mt(file_name_testpattern, 2);
	// image: house
	testRunRecursiveBF_AVX2_mt(file_name_house, 2);
	// image: testGirl
	testRunRecursiveBF_AVX2_mt(file_name_testGirl, 2);

	////////////////////////
	cout << "\n\nOptimized AVX2 4x multithreading, single stage (non-pipelined), memory aligned";
	// image: testpattern
	testRunRecursiveBF_AVX2_mt(file_name_testpattern, 4);
	// image: house
	testRunRecursiveBF_AVX2_mt(file_name_house, 4);
	// image: testGirl
	testRunRecursiveBF_AVX2_mt(file_name_testGirl, 4);

	////////////////////////
	cout << "\n\nOptimized AVX2 4x2 thread pipelined 2 stages, memory aligned";
	// image: testpattern
	testRunRecursiveBF_AVX2_Pipelined(file_name_testpattern, 4);
	// image: house
	testRunRecursiveBF_AVX2_Pipelined(file_name_house, 4);
	// image: testGirl
	testRunRecursiveBF_AVX2_Pipelined(file_name_testGirl, 4);

	cout << "\nFinish";
	cin.get();

    return 0;
}

