#pragma once

// Optimized SSE2 implementation of Recursive Bilateral Filter
// 

#include <future>

#define RBF_MAX_THREADS 8
#define STAGE_BUFFER_COUNT 3

class CRBFilterSSE2
{
	int				m_reserved_width = 0;
	int				m_reserved_height = 0;
	int				m_thread_count = 0;
	bool			m_pipelined = false;

	float			m_sigma_spatial = 0.f;
	float			m_sigma_range = 0.f;
	float			m_inv_alpha_f = 0.f;
	float*			m_range_table = nullptr;

	int				m_filter_counter = 0; // used in pipelined mode
	unsigned char*	m_stage_buffer[STAGE_BUFFER_COUNT] = { nullptr }; // size width * height * 4, 2nd one null if not pipelined
	float**			m_h_line_cache = nullptr; // single line cache for horizontal filter pass, one per thread
	float**			m_v_line_cache = nullptr; // if not pipelined mode, this is equal to 'm_h_line_cache'
	unsigned char*	m_out_buffer[STAGE_BUFFER_COUNT] = { nullptr }; // used for keeping track of current output buffer in pipelined mode 
	int				m_image_width = 0; // cache of sizes for pipelined mode
	int				m_image_height = 0;
	int				m_image_pitch = 0;

	std::future<void> m_horizontal_tasks[RBF_MAX_THREADS];
	std::future<void> m_vertical_tasks[RBF_MAX_THREADS];

	// core filter functions
	void horizontalFilter(int thread_index, const unsigned char* img_src, unsigned char* img_dst, int width, int height, int pitch);
	void verticalFilter(int thread_index, const unsigned char* img_src, unsigned char* img_dst, int width, int height, int pitch);

public:

	CRBFilterSSE2();
	~CRBFilterSSE2();

	// 'sigma_spatial' - unlike the original implementation of Recursive Bilateral Filter, 
	// the value if sigma_spatial is not influence by image width/height.
	// In this implementation, sigma_spatial is assumed over image width 255, height 255
	void setSigma(float sigma_spatial, float sigma_range);

	// Source and destination images are assumed to be 4 component
	// 'width' - maximum image width
	// 'height' - maximum image height
	// 'thread_count' - total thread count to use for each filter stage (horizontal and vertical), recommended thread count = 4
	// 'pipelined' - if true, then horizontal and vertical filter passes are split into separate stages,
	// where each stage uses 'thread_count' of threads (so basically double)
	// Return true if successful, had very basic error checking
	bool initialize(int width, int height, int thread_count = 1, bool pipelined = false);
	
	// de-initialize, free memory
	void release();

	// synchronous filter function, returns only when everything finished, goes faster if there's multiple threads
	// initialize() and setSigma() should be called before this
	// 'out_data' - output image buffer, assumes 4 byte per pixel
	// 'in_data' - input image buffer, assumes 4 byte per pixel
	// 'width' - width of both input and output buffers, must be same for both
	// 'height' - height of both input and output buffers, must be same for both
	// 'pitch' - row size in bytes, must be same for both buffers (ideally, this should be divisible by 16)
	// return false if failed for some reason
	bool filter(unsigned char* out_data, const unsigned char* in_data, int width, int height, int pitch);

	// asynchronous, pipelined filter function
	// pipeline consists of 2 stages, one for horizontal filter, other for vertical filter
	// this is useful for video filtering where 1-2 frame delay is acceptable
	// for simplicity of this sample implementation, input and output data buffers must remain valid until filtering is finished
	// since it's 2 stage pipeline, consecutive calls should submit alternating buffers (2 sets of input and output buffers)
	// This function blocks until 1st stage finishes from previous call
	bool filterPipePush(unsigned char* out_data, const unsigned char* in_data, int width, int height, int pitch);
	// this function blocks until both stages finished all processing
	// it should always be used to get last frame
	void filterPipeFlush();
};