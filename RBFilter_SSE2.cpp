#include "stdafx.h"
#include "RBFilter_SSE2.h"
#include <math.h>
#include <malloc.h>
#include <new>  
#include <emmintrin.h>
#include <tmmintrin.h>
#include <thread>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>


#define MAX_RANGE_TABLE_SIZE 255
#define ALIGN_SIZE 16

// only 1 of following 2 should be defined
#define EDGE_COLOR_USE_MAXIMUM
//#define EDGE_COLOR_USE_ADDITION

// if EDGE_COLOR_USE_MAXIMUM is defined, then edge color detection works by calculating
// maximum difference among 3 components (RGB) of 2 colors, which tends to result in lower differences (since only largest among 3 is selected)
// if EDGE_COLOR_USE_ADDITION is defined, then edge color detection works by calculating
// sum of all 3 components, while enforcing 255 maximum. This method is much more sensitive to small differences 

#if defined(EDGE_COLOR_USE_MAXIMUM) && defined(EDGE_COLOR_USE_ADDITION)
#error Only 1 of those can be defined
#endif

#if !defined(EDGE_COLOR_USE_MAXIMUM) && !defined(EDGE_COLOR_USE_ADDITION)
#error 1 of those must be defined
#endif

CRBFilterSSE2::CRBFilterSSE2()
{
	m_range_table = new float[MAX_RANGE_TABLE_SIZE + 1];
	memset(m_range_table, 0, (MAX_RANGE_TABLE_SIZE + 1) * sizeof(float));
}

CRBFilterSSE2::~CRBFilterSSE2()
{
	release();

	delete[] m_range_table;
}

bool CRBFilterSSE2::initialize(int width, int height, int thread_count, bool pipelined)
{
	// basic sanity check, not strict
	if (width < 16 || width > 10000)
		return false;

	if (height < 2 || height > 10000)
		return false;

	if (thread_count < 1 || thread_count > RBF_MAX_THREADS)
		return false;
	
	release();

	// round width up to nearest ALIGN_SIZE * thread_count
	int round_up = (ALIGN_SIZE / 4) * thread_count;
	if (width % round_up)
	{
		width += round_up - width % round_up;
	}
	m_reserved_width = width;
	m_reserved_height = height;
	m_thread_count = thread_count;

//	m_stage_buffer[0] = (unsigned char*)_aligned_malloc(m_reserved_width * m_reserved_height * 4, ALIGN_SIZE);
	m_stage_buffer[0] = (unsigned char*)aligned_alloc(ALIGN_SIZE, m_reserved_width * m_reserved_height * 4);
	if (!m_stage_buffer[0])
		return false;

	if (pipelined)
	{
		for (int i = 1; i < STAGE_BUFFER_COUNT; i++)
		{
//			m_stage_buffer[i] = (unsigned char*)_aligned_malloc(m_reserved_width * m_reserved_height * 4, ALIGN_SIZE);
			m_stage_buffer[i] = (unsigned char*)aligned_alloc(ALIGN_SIZE, m_reserved_width * m_reserved_height * 4);
			if (!m_stage_buffer[i])
				return false;
		}
	}

	m_h_line_cache = new (std::nothrow) float*[m_thread_count];
	if (!m_h_line_cache)
		return false;

	// zero just in case
	for (int i = 0; i < m_thread_count; i++)
		m_h_line_cache[i] = nullptr;

	for (int i = 0; i < m_thread_count; i++)
	{
//		m_h_line_cache[i] = (float*)_aligned_malloc(m_reserved_width * 12 * sizeof(float) , ALIGN_SIZE);
		m_h_line_cache[i] = (float*)aligned_alloc(ALIGN_SIZE, m_reserved_width * 12 * sizeof(float));
		if (!m_h_line_cache[i])
			return false;
	}

//	if (m_pipelined)
	{
		m_v_line_cache = new (std::nothrow) float*[m_thread_count];
		if (!m_v_line_cache)
			return false;

		for (int i = 0; i < m_thread_count; i++)
			m_v_line_cache[i] = nullptr;

		for (int i = 0; i < m_thread_count; i++)
		{
//			m_v_line_cache[i] = (float*)_aligned_malloc((m_reserved_width * 8 * sizeof(float)) / m_thread_count, ALIGN_SIZE);
            m_v_line_cache[i] = (float*)aligned_alloc(ALIGN_SIZE, (m_reserved_width * 8 * sizeof(float)) / m_thread_count);
			if (!m_v_line_cache[i])
				return false;
		}
	}


	return true;
}

void CRBFilterSSE2::release()
{
	for (int i = 0; i < STAGE_BUFFER_COUNT; i++)
	{
		if (m_stage_buffer[i])
		{
			free(m_stage_buffer[i]);
			m_stage_buffer[i] = nullptr;
		}
	}

	if (m_h_line_cache)
	{
		for (int i = 0; i < m_thread_count; i++)
		{
			if (m_h_line_cache[i])
				free(m_h_line_cache[i]);
		}
		delete[] m_h_line_cache;
		m_h_line_cache = nullptr;
	}

//	if (m_pipelined)
	{
		for (int i = 0; i < m_thread_count; i++)
		{
			if (m_v_line_cache[i])
				free(m_v_line_cache[i]);
		}
		delete[] m_v_line_cache;
	}
	m_v_line_cache = nullptr;

	m_reserved_width = 0;
	m_reserved_height = 0;
	m_thread_count = 0;
	m_pipelined = false;
	m_filter_counter = 0;
}

void CRBFilterSSE2::setSigma(float sigma_spatial, float sigma_range)
{
	if (m_sigma_spatial != sigma_spatial || m_sigma_range != sigma_range)
	{
		m_sigma_spatial = sigma_spatial;
		m_sigma_range = sigma_range;

		double alpha_f = (exp(-sqrt(2.0) / (sigma_spatial * 255.0)));
		m_inv_alpha_f = (float)(1.0 - alpha_f);
		double inv_sigma_range = 1.0 / (sigma_range * MAX_RANGE_TABLE_SIZE);
		{
			double ii = 0.f;
			for (int i = 0; i <= MAX_RANGE_TABLE_SIZE; i++, ii -= 1.0)
			{
				m_range_table[i] = (float)(alpha_f * exp(ii * inv_sigma_range));
			}
		}
	}
}

// example of edge color difference calculation from original implementation
// idea is to fit maximum edge color difference as single number in 0-255 range
// colors are added then 2 components are scaled 4x while 1 complement is scaled 2x
// this means 1 of the components is more dominant 

//int getDiffFactor(const unsigned char* color1, const unsigned char* color2)
//{
//	int c1 = abs(color1[0] - color2[0]);
//	int c2 = abs(color1[1] - color2[1]);
//	int c3 = abs(color1[2] - color2[2]);
//
//	return ((c1 + c3) >> 2) + (c2 >> 1);
//}


inline void getDiffFactor3x(__m128i pix4, __m128i pix4p, __m128i* diff4x)
{
	static __m128i byte_mask = _mm_set1_epi32(255);

	// get absolute difference for each component per pixel
	__m128i diff = _mm_sub_epi8(_mm_max_epu8(pix4, pix4p), _mm_min_epu8(pix4, pix4p));

#ifdef EDGE_COLOR_USE_MAXIMUM
	// get maximum of 3 components
	__m128i diff_shift1 = _mm_srli_epi32(diff, 8); // 2nd component
	diff = _mm_max_epu8(diff, diff_shift1);
	diff_shift1 = _mm_srli_epi32(diff_shift1, 8); // 3rd component
	diff = _mm_max_epu8(diff, diff_shift1);
	// skip alpha component
	diff = _mm_and_si128(diff, byte_mask); // zero out all but 1st byte
#endif

#ifdef EDGE_COLOR_USE_ADDITION
	// add all component differences and saturate 
	__m128i diff_shift1 = _mm_srli_epi32(diff, 8); // 2nd component
	diff = _mm_adds_epu8(diff, diff_shift1);
	diff_shift1 = _mm_srli_epi32(diff_shift1, 8); // 3rd component
	diff = _mm_adds_epu8(diff, diff_shift1);
	diff = _mm_and_si128(diff, byte_mask); // zero out all but 1st byte
#endif
	_mm_store_si128(diff4x, diff);
}


void CRBFilterSSE2::horizontalFilter(int thread_index, const unsigned char* img_src, unsigned char* img_dst, int width, int height, int pitch)
{
	int height_segment = height / m_thread_count;
	int buffer_offset = thread_index * height_segment * pitch;
	img_src += buffer_offset;
	img_dst += buffer_offset;

	if (thread_index + 1 == m_thread_count) // last segment should account for uneven height
		height_segment += height % m_thread_count;

	float* line_cache = m_h_line_cache[thread_index];
	const float* range_table = m_range_table;

	__m128 inv_alpha = _mm_set_ps1(m_inv_alpha_f);
	__m128 half_value = _mm_set_ps1(0.5f);
	__m128i mask_pack = _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
	__m128i mask_unpack = _mm_setr_epi8(12, -1, -1, -1, 13, -1, -1, -1, 14, -1, -1, -1, 15, -1, -1, -1);

	// used to store maximum difference between 2 pixels
//	__declspec(align(16)) long color_diff[4];
	alignas(16) int color_diff[4];

	for (int y = 0; y < height_segment; y++)
	{
		//////////////////////
		// right to left pass, results of this pass get stored in 'line_cache'
		{
			int pixels_left = width - 1;

			// get end of line buffer
			float* line_buffer = line_cache + pixels_left * 12;

			///////
			// handle last pixel in row separately as special case
			{
				const unsigned char* last_src = img_src + (y + 1) * pitch - 4;

				// result color
				line_buffer[8] = (float)last_src[0];
				line_buffer[9] = (float)last_src[1];
				line_buffer[10] = (float)last_src[2];
				line_buffer[11] = (float)last_src[3];

				// premultiplied source
				// caching pre-multiplied allows saving 1 multiply operation in 2nd pass loop, not a big difference
				line_buffer[4] = m_inv_alpha_f * line_buffer[8];
				line_buffer[5] = m_inv_alpha_f * line_buffer[9];
				line_buffer[6] = m_inv_alpha_f * line_buffer[10];
				line_buffer[7] = m_inv_alpha_f * line_buffer[11];
			}

			// "previous" pixel color
			__m128 pixel_prev = _mm_load_ps(line_buffer + 8);
			// "previous" pixel factor
			__m128 alpha_f_prev4 = _mm_set_ps1(1.f);

			///////
			// handle most middle pixels in 16 byte intervals using xmm registers
			// process 4x pixels at a time
			int buffer_inc = y * pitch + (pixels_left - 1) * 4 - 16;
			const __m128i* src_4xCur = (const __m128i*)(img_src + buffer_inc);
			const __m128i* src_4xPrev = (const __m128i*)(img_src + buffer_inc + 4);
			while (pixels_left > 0) // outer loop 4x pixel
			{
				// load 4x pixel, may read backward past start of buffer, but it's OK since that extra data won't be used
				__m128i pix4 = _mm_loadu_si128(src_4xCur--);
				__m128i pix4p = _mm_loadu_si128(src_4xPrev--);

				// get color differences
				getDiffFactor3x(pix4, pix4p, (__m128i*)color_diff);

				for (int i = 3; i >= 0 && pixels_left-- > 0; i--) // inner loop
				{
					float alpha_f = range_table[color_diff[i]];
					__m128 alpha_f_4x = _mm_set_ps1(alpha_f);

					// cache weights for next filter pass
					line_buffer -= 12;
					_mm_store_ps(line_buffer, alpha_f_4x);

					// color factor
					alpha_f_prev4 = _mm_mul_ps(alpha_f_prev4, alpha_f_4x);
					alpha_f_prev4 = _mm_add_ps(alpha_f_prev4, inv_alpha);

					// unpack current source pixel
					__m128i pix1 = _mm_shuffle_epi8(pix4, mask_unpack); // extracts 1 pixel components from BYTE to DWORD
					pix4 = _mm_slli_si128(pix4, 4); // shift left so next loop unpacks next pixel data 
					__m128 pixel_F = _mm_cvtepi32_ps(pix1); // convert to floats
					

					// apply color filter
					pixel_F = _mm_mul_ps(pixel_F, inv_alpha);
					_mm_store_ps(line_buffer + 4, pixel_F); // cache pre-multiplied source color for next filter pass
					alpha_f_4x = _mm_mul_ps(pixel_prev, alpha_f_4x);
					pixel_F = _mm_add_ps(pixel_F, alpha_f_4x);

					// store current color as previous for next cycle
					pixel_prev = pixel_F;

					// calculate final color
					pixel_F = _mm_div_ps(pixel_F, alpha_f_prev4);

					// cache filtered color for next filter pass
					_mm_store_ps(line_buffer + 8, pixel_F);
				}
			}
		}

		//////////////////////
		// left to right pass
		{
			int pixels_left = width - 1;

			// process 4x pixels at a time
			int buffer_inc = y * pitch;
			const __m128i* src_4xCur = (const __m128i*)(img_src + buffer_inc + 4); // shifted by 1 pixel
			const __m128i* src_4xPrev = (const __m128i*)(img_src + buffer_inc);

			// use float type only to enable 4 byte write using MOVSS
			float* out_result = (float*)(img_dst + buffer_inc + 4); // start at 2nd pixel from left

			const float* line_buffer = line_cache;

			///////
			// handle first pixel in row separately as special case
			{
				unsigned char* first_dst = img_dst + buffer_inc;
				// average new pixel with one already in output
				// source color was pre-multipled, so get original
				float inv_factor = 1.f / m_inv_alpha_f;
				first_dst[0] = (unsigned char)((line_buffer[4] * inv_factor + line_buffer[8]) * 0.5f);
				first_dst[1] = (unsigned char)((line_buffer[5] * inv_factor + line_buffer[9]) * 0.5f);
				first_dst[2] = (unsigned char)((line_buffer[6] * inv_factor + line_buffer[10]) * 0.5f);
				first_dst[3] = (unsigned char)((line_buffer[7] * inv_factor + line_buffer[11]) * 0.5f);
			}

			// initialize "previous pixel" with 4 components of last row pixel
			__m128 pixel_prev = _mm_load_ps(line_buffer + 8);
			line_buffer += 12;
			__m128 alpha_f_prev4 = _mm_set_ps1(1.f);


			///////
			// handle most pixels in 16 byte intervals using xmm registers
			while (pixels_left > 0) // outer loop 4x pixel
			{
				for (int i = 0; i <= 3 && pixels_left-- > 0; i++) // inner loop
				{
					// load cached factor
					__m128 alpha_f_4x = _mm_load_ps(line_buffer);
					line_buffer += 12;

					// color factor
					alpha_f_prev4 = _mm_mul_ps(alpha_f_prev4, alpha_f_4x);
					alpha_f_prev4 = _mm_add_ps(alpha_f_prev4, inv_alpha);

					// load current source pixel, pre-multiplied
					__m128 pixel_F = _mm_load_ps(line_buffer + 4);


					// apply color filter
					alpha_f_4x = _mm_mul_ps(pixel_prev, alpha_f_4x);
					pixel_F = _mm_add_ps(pixel_F, alpha_f_4x);

					// store current color as previous for next cycle
					pixel_prev = pixel_F;

					// calculate final color
					pixel_F = _mm_div_ps(pixel_F, alpha_f_prev4);

					// average this result with result from previous pass
					__m128 prev_pix4 = _mm_load_ps(line_buffer + 8);

					pixel_F = _mm_add_ps(pixel_F, prev_pix4);
					pixel_F = _mm_mul_ps(pixel_F, half_value);

					// pack float pixel into byte pixel
					__m128i pixB = _mm_cvtps_epi32(pixel_F); // convert to integer
					pixB = _mm_shuffle_epi8(pixB, mask_pack);
					_mm_store_ss(out_result++, _mm_castsi128_ps(pixB));

				}
			}
		}
	}
}


void CRBFilterSSE2::verticalFilter(int thread_index, const unsigned char* img_src, unsigned char* img_dst, int width, int height, int pitch)
{
	int width_segment = width / m_thread_count;
	// make sure width segments round to 16 byte boundary except for last one
	width_segment -= width_segment % 4;
	int start_offset = width_segment * thread_index;
	if (thread_index == m_thread_count - 1) // last one
		width_segment = width - start_offset;

	int width4 = width_segment / 4;

	// adjust img buffer starting positions
	img_src += start_offset * 4;
	img_dst += start_offset * 4;

	float* line_cache = m_v_line_cache[thread_index];
	const float* range_table = m_range_table;

	__m128 inv_alpha = _mm_set_ps1(m_inv_alpha_f);
	__m128 half_value = _mm_set_ps1(0.5f);
	__m128i mask_pack = _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
	__m128i mask_unpack = _mm_setr_epi8(0, -1, -1, -1, 1, -1, -1, -1, 2, -1, -1, -1, 3, -1, -1, -1);

	// used to store maximum difference between 2 pixels
//	__declspec(align(16)) long color_diff[4];
	alignas(16) int color_diff[4];

	/////////////////
	// Bottom to top pass first
	{
		// last line processed separately since no previous
		{
			unsigned char* dst_line = img_dst + (height - 1) * pitch;
			const unsigned char* src_line = img_src + (height - 1) * pitch;
			float* line_buffer = line_cache;

			memcpy(dst_line, src_line, width_segment * 4); // copy last line

			// initialize line cache
			for (int x = 0; x < width_segment; x++)
			{
				// set factor to 1
				line_buffer[0] = 1.f;
				line_buffer[1] = 1.f;
				line_buffer[2] = 1.f;
				line_buffer[3] = 1.f;

				// set result color
				line_buffer[4] = (float)src_line[0];
				line_buffer[5] = (float)src_line[1];
				line_buffer[6] = (float)src_line[2];
				line_buffer[7] = (float)src_line[3];

				src_line += 4;
				line_buffer += 8;
			}
		}

		// process other lines
		for (int y = height - 2; y >= 0; y--)
		{
			float* dst_line = (float*)(img_dst + y * pitch);
			float* line_buffer = line_cache;

			__m128i* src_4xCur = (__m128i*)(img_src + y * pitch);
			__m128i* src_4xPrev = (__m128i*)(img_src + (y + 1) * pitch);

			int pixels_left = width_segment;
			while (pixels_left > 0)
			{
				// may read past end of buffer, but that data won't be used
				__m128i pix4 = _mm_loadu_si128(src_4xCur++); // load 4x pixel
				__m128i pix4p = _mm_loadu_si128(src_4xPrev++);

				// get color differences
				getDiffFactor3x(pix4, pix4p, (__m128i*)color_diff);

				for (int i = 0; i < 4 && pixels_left-- > 0; i++) // inner loop
				{
					float alpha_f = range_table[color_diff[i]];
					__m128 alpha_f_4x = _mm_set_ps1(alpha_f);

					// load previous line color factor
					__m128 alpha_f_prev4 = _mm_load_ps(line_buffer);
					// load previous line color
					__m128 pixel_prev = _mm_load_ps(line_buffer + 4);

					// color factor
					alpha_f_prev4 = _mm_mul_ps(alpha_f_prev4, alpha_f_4x);
					alpha_f_prev4 = _mm_add_ps(alpha_f_prev4, inv_alpha);

					// unpack current source pixel
					__m128i pix1 = _mm_shuffle_epi8(pix4, mask_unpack);
					pix4 = _mm_srli_si128(pix4, 4); // shift right
					__m128 pixel_F = _mm_cvtepi32_ps(pix1); // convert to floats
					

					// apply color filter
					pixel_F = _mm_mul_ps(pixel_F, inv_alpha);
					alpha_f_4x = _mm_mul_ps(pixel_prev, alpha_f_4x);
					pixel_F = _mm_add_ps(pixel_F, alpha_f_4x);

					// store current factor and color as previous for next cycle
					_mm_store_ps(line_buffer, alpha_f_prev4);
					_mm_store_ps(line_buffer + 4, pixel_F);
					line_buffer += 8;

					// calculate final color
					pixel_F = _mm_div_ps(pixel_F, alpha_f_prev4);

					// pack float pixel into byte pixel
					__m128i pixB = _mm_cvtps_epi32(pixel_F); // convert to integer
					pixB = _mm_shuffle_epi8(pixB, mask_pack);
					_mm_store_ss(dst_line++, _mm_castsi128_ps(pixB));
				}
			}
		}
	}

	/////////////////
	// Top to bottom pass last
	{
		mask_pack = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 4, 8, 12);

		// first line handled separately since no previous
		{
			unsigned char* dst_line = img_dst;
			const unsigned char* src_line = img_src;
			float* line_buffer = line_cache;

			for (int x = 0; x < width_segment; x++)
			{
				// average ccurrent destanation color with current source
				dst_line[0] = (dst_line[0] + src_line[0]) / 2;
				dst_line[1] = (dst_line[1] + src_line[1]) / 2;
				dst_line[2] = (dst_line[2] + src_line[2]) / 2;
				dst_line[3] = (dst_line[3] + src_line[3]) / 2;

				// set factor to 1
				line_buffer[0] = 1.f;
				line_buffer[1] = 1.f;
				line_buffer[2] = 1.f;
				line_buffer[3] = 1.f;

				// set result color
				line_buffer[4] = (float)src_line[0];
				line_buffer[5] = (float)src_line[1];
				line_buffer[6] = (float)src_line[2];
				line_buffer[7] = (float)src_line[3];

				dst_line += 4;
				src_line += 4;
				line_buffer += 8;
			}
		}

		// process other lines
		for (int y = 1; y < height; y++)
		{
			//	const unsigned char* src_line = img_src + y * pitch;
			float* line_buffer = line_cache;

			__m128i* src_4xCur = (__m128i*)(img_src + y * pitch);
			__m128i* src_4xPrev = (__m128i*)(img_src + (y - 1) * pitch);
			__m128i* dst_4x = (__m128i*)(img_dst + y * pitch);

			for (int x = 0; x < width4; x++)
			{
				// get color difference
				__m128i pix4 = _mm_loadu_si128(src_4xCur++); // load 4x pixel
				__m128i pix4p = _mm_loadu_si128(src_4xPrev++);

				// get color differences
				getDiffFactor3x(pix4, pix4p, (__m128i*)color_diff);

				__m128i out_pix4;
				for (int i = 0; i < 4; i++) // inner loop
				{
					float alpha_f = range_table[color_diff[i]];
					__m128 alpha_f_4x = _mm_set_ps1(alpha_f);

					// load previous line color factor
					__m128 alpha_f_prev4 = _mm_load_ps(line_buffer);
					// load previous line color
					__m128 pixel_prev = _mm_load_ps(line_buffer + 4);

					// color factor
					//	alpha_f_prev = m_inv_alpha_f + alpha_f * alpha_f_prev;
					alpha_f_prev4 = _mm_mul_ps(alpha_f_prev4, alpha_f_4x);
					alpha_f_prev4 = _mm_add_ps(alpha_f_prev4, inv_alpha);

					// unpack current source pixel
					__m128i pix1 = _mm_shuffle_epi8(pix4, mask_unpack);
					pix4 = _mm_srli_si128(pix4, 4); // shift right
					__m128 pixel_F = _mm_cvtepi32_ps(pix1); // convert to floats

					// apply color filter
					pixel_F = _mm_mul_ps(pixel_F, inv_alpha);
					alpha_f_4x = _mm_mul_ps(pixel_prev, alpha_f_4x);
					pixel_F = _mm_add_ps(pixel_F, alpha_f_4x);

					// store current factor and color as previous for next cycle
					_mm_store_ps(line_buffer, alpha_f_prev4);
					_mm_store_ps(line_buffer + 4, pixel_F);
					line_buffer += 8;

					// calculate final color
					pixel_F = _mm_div_ps(pixel_F, alpha_f_prev4);

					// pack float pixel into byte pixel
					__m128i pixB = _mm_cvtps_epi32(pixel_F); // convert to integer
					pixB = _mm_shuffle_epi8(pixB, mask_pack);

					out_pix4 = _mm_srli_si128(out_pix4, 4); // shift 
					out_pix4 = _mm_or_si128(out_pix4, pixB);

				}

				// average result 4x pixel with what is already in destination
				__m128i dst4 = _mm_loadu_si128(dst_4x);
				out_pix4 = _mm_avg_epu8(out_pix4, dst4);
				_mm_storeu_si128(dst_4x++, out_pix4); // store 4x pixel
			}

			// have to handle leftover 1-3 pixels if last width segment isn't divisble by 4
			if (width_segment % 4)
			{
				// this should be avoided by having image buffers with pitch divisible by 16
			}
		}
	}

}

bool CRBFilterSSE2::filter(unsigned char* out_data, const unsigned char* in_data, int width, int height, int pitch)
{
	// basic error checking
	if (!m_stage_buffer[0])
		return false;

	if (width < 16 || width > m_reserved_width)
		return false;

	if (height < 16 || height > m_reserved_height)
		return false;

	if (pitch < width * 4)
		return false;

	if (!out_data || !in_data)
		return false;

	if (m_inv_alpha_f == 0.f)
		return false;

	int thread_count_adjusted = m_thread_count - 1;

	////////////////////////////////////////////// 
	// horizontal filter divided in threads
	for (int i = 0; i < thread_count_adjusted; i++)
	{
		m_horizontal_tasks[i] = std::async(std::launch::async, &CRBFilterSSE2::horizontalFilter, this, i, in_data, m_stage_buffer[0], width, height, pitch);
	}

	// use this thread for last segment
	horizontalFilter(thread_count_adjusted, in_data, m_stage_buffer[0], width, height, pitch);

	// wait for result
	for (int i = 0; i < thread_count_adjusted; i++)
	{
		m_horizontal_tasks[i].get();
	}

	/////////////////////////////////////////////
	// vertical filter divided in threads
	for (int i = 0; i < thread_count_adjusted; i++)
	{
		m_vertical_tasks[i] = std::async(std::launch::async, &CRBFilterSSE2::verticalFilter, this, i, m_stage_buffer[0], out_data, width, height, pitch);
	}

	// use this thread for last segment
	verticalFilter(thread_count_adjusted, m_stage_buffer[0], out_data, width, height, pitch);

	// wait for result
	for (int i = 0; i < thread_count_adjusted; i++)
	{
		m_vertical_tasks[i].get();
	}

	return true;
}

bool CRBFilterSSE2::filterPipePush(unsigned char* out_data, const unsigned char* in_data, int width, int height, int pitch)
{
	// basic error checking
	if (!m_stage_buffer[0])
		return false;

	if (width < 16 || width > m_reserved_width)
		return false;

	if (height < 16 || height > m_reserved_height)
		return false;

	if (pitch < width * 4)
		return false;

	if (m_inv_alpha_f == 0.f)
		return false;

	m_image_width = width;
	m_image_height = height;
	m_image_pitch = pitch;

	// block until last frame finished 1st stage
	for (int i = 0; i < m_thread_count; i++)
	{
		if (m_horizontal_tasks[i].valid())
			m_horizontal_tasks[i].get();
	}

	int previous_stage_index = (m_filter_counter - 1) % STAGE_BUFFER_COUNT;
	int current_stage_index = m_filter_counter % STAGE_BUFFER_COUNT;
	m_filter_counter++;
	m_out_buffer[current_stage_index] = out_data;

	// start new horizontal stage
	if (in_data)
	{
		// start first stage for current frame
		for (int i = 0; i < m_thread_count; i++)
		{
			m_horizontal_tasks[i] = std::async(std::launch::async, &CRBFilterSSE2::horizontalFilter, this, i, in_data, m_stage_buffer[current_stage_index], width, height, pitch);
		}
	}

	// block until last frame finished 2nd stage
	for (int i = 0; i < m_thread_count; i++)
	{
		if (m_vertical_tasks[i].valid())
			m_vertical_tasks[i].get();
	}

	// start new vertical stage based on result of previous stage
	if (previous_stage_index >= 0 && m_out_buffer[previous_stage_index])
	{
		// start first stage for current frame
		for (int i = 0; i < m_thread_count; i++)
		{
			m_vertical_tasks[i] = std::async(std::launch::async, &CRBFilterSSE2::verticalFilter, this, i, m_stage_buffer[previous_stage_index], m_out_buffer[previous_stage_index], width, height, pitch);
		}
	}

	return true;
}

void CRBFilterSSE2::filterPipeFlush()
{
	filterPipePush(nullptr, nullptr, m_image_width, m_image_height, m_image_pitch);

	if (m_filter_counter > 0)
	{
		for (int i = 0; i < m_thread_count; i++)
		{
			if(m_vertical_tasks[i].valid())
				m_vertical_tasks[i].get();
		}
	}

	m_filter_counter = 0;
}