#include "stdafx.h"
#include "RBFilter_AVX2.h"
#include <math.h>
#include <malloc.h>
#include <new>  
#include <emmintrin.h>
#include <tmmintrin.h>
#include <immintrin.h>
#include <smmintrin.h>
#include <thread>


#define MAX_RANGE_TABLE_SIZE 255
#define ALIGN_SIZE 32

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

CRBFilterAVX2::CRBFilterAVX2()
{
	m_range_table = new float[MAX_RANGE_TABLE_SIZE + 1];
	memset(m_range_table, 0, (MAX_RANGE_TABLE_SIZE + 1) * sizeof(float));
}

CRBFilterAVX2::~CRBFilterAVX2()
{
	release();

	delete[] m_range_table;
}

bool CRBFilterAVX2::initialize(int width, int height, int thread_count, bool pipelined)
{
	// basic sanity check, not strict
	if (width < 16 || width > 10000)
		return false;

	if (height < 2 || height > 10000)
		return false;

	if (thread_count < 1 || thread_count > RBF_MAX_THREADS)
		return false;
	
	release();

	m_thread_count = thread_count;

	// round height to nearest even number
	if (height & 1)
		height++;

	m_reserved_width = getOptimalPitch(width) / 4;
	m_reserved_height = height;
	

	m_stage_buffer[0] = (unsigned char*)_aligned_malloc(m_reserved_width * m_reserved_height * 4, ALIGN_SIZE);
	if (!m_stage_buffer[0])
		return false;

	if (pipelined)
	{
		for (int i = 1; i < STAGE_BUFFER_COUNT; i++)
		{
			m_stage_buffer[i] = (unsigned char*)_aligned_malloc(m_reserved_width * m_reserved_height * 4, ALIGN_SIZE);
			if (!m_stage_buffer[i])
				return false;
		}
	}

	/////////////////
	m_h_line_cache = new (std::nothrow) float*[m_thread_count];
	if (!m_h_line_cache)
		return false;

	// zero just in case
	for (int i = 0; i < m_thread_count; i++)
		m_h_line_cache[i] = nullptr;

	for (int i = 0; i < m_thread_count; i++)
	{
		m_h_line_cache[i] = (float*)_aligned_malloc(m_reserved_width * 12 * sizeof(float) * 2 + 128, ALIGN_SIZE);
		if (!m_h_line_cache[i])
			return false;

		// 1st 8 bytes of line cache should remain constant zero
		memset(m_h_line_cache[i], 0, 8 * sizeof(float));
	}

	////////////////
	m_v_line_cache = new (std::nothrow) float*[m_thread_count];
	if (!m_v_line_cache)
		return false;

	for (int i = 0; i < m_thread_count; i++)
		m_v_line_cache[i] = nullptr;

	int v_line_size = (m_reserved_width * 16 * sizeof(float)) / m_thread_count;
	for (int i = 0; i < m_thread_count; i++)
	{
		m_v_line_cache[i] = (float*)_aligned_malloc(v_line_size, ALIGN_SIZE);
		if (!m_v_line_cache[i])
			return false;
	}

	return true;
}

void CRBFilterAVX2::release()
{
	for (int i = 0; i < STAGE_BUFFER_COUNT; i++)
	{
		if (m_stage_buffer[i])
		{
			_aligned_free(m_stage_buffer[i]);
			m_stage_buffer[i] = nullptr;
		}
	}

	if (m_h_line_cache)
	{
		for (int i = 0; i < m_thread_count; i++)
		{
			if (m_h_line_cache[i])
				_aligned_free(m_h_line_cache[i]);
		}
		delete[] m_h_line_cache;
		m_h_line_cache = nullptr;
	}

	if (m_v_line_cache)
	{
		for (int i = 0; i < m_thread_count; i++)
		{
			if (m_v_line_cache[i])
				_aligned_free(m_v_line_cache[i]);
		}
		delete[] m_v_line_cache;
		m_v_line_cache = nullptr;
	}

	m_reserved_width = 0;
	m_reserved_height = 0;
	m_thread_count = 0;
	m_pipelined = false;
	m_filter_counter = 0;
}

int CRBFilterAVX2::getOptimalPitch(int width) const
{
	width *= 4;

	int round_up = ALIGN_SIZE * m_thread_count;
	if (width % round_up)
	{
		width += round_up - width % round_up;
	}

	return width;
}

void CRBFilterAVX2::setSigma(float sigma_spatial, float sigma_range)
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


inline void getDiffFactor3x(__m256i pix8, __m256i pix8p, __m256i* diff8x)
{
	__m256i byte_mask = _mm256_set1_epi32(255);

	// get absolute difference for each component per pixel
	__m256i diff = _mm256_sub_epi8(_mm256_max_epu8(pix8, pix8p), _mm256_min_epu8(pix8, pix8p));

#ifdef EDGE_COLOR_USE_MAXIMUM
	// get maximum of 3 components
	__m256i diff_shift1 = _mm256_srli_epi32(diff, 8); // 2nd component
	diff = _mm256_max_epu8(diff, diff_shift1);
	diff_shift1 = _mm256_srli_epi32(diff_shift1, 8); // 3rd component
	diff = _mm256_max_epu8(diff, diff_shift1);
	// skip alpha component
	diff = _mm256_and_si256(diff, byte_mask); // zero out all but 1st byte
#endif

#ifdef EDGE_COLOR_USE_ADDITION
	// add all component differences and saturate 
	__m256i diff_shift1 = _mm256_srli_epi32(diff, 8); // 2nd component
	diff = _mm256_adds_epu8(diff, diff_shift1);
	diff_shift1 = _mm256_srli_epi32(diff_shift1, 8); // 3rd component
	diff = _mm256_adds_epu8(diff, diff_shift1);
	diff = _mm256_and_si256(diff, byte_mask); // zero out all but 1st byte
#endif

	_mm256_store_si256(diff8x, diff);
}

void CRBFilterAVX2::horizontalFilter(int thread_index, const unsigned char* img_src, unsigned char* img_dst, int width, int height, int pitch)
{
	// force height segments to be even cause this filter processes 2 lines at a time
	int height_segment = (height / m_thread_count) & (~1); 
	int buffer_offset = thread_index * height_segment * pitch;
	img_src += buffer_offset;
	img_dst += buffer_offset;

	int width32 = pitch / 32;

	// last segment should account for uneven height
	// since reserve buffer height is rounded up to even number, it's OK if source is uneven
	// but that assumes hozitonal filter output buffer is the reservered buffer, or that destination is rounded up to even number
	if (thread_index + 1 == m_thread_count) 
		height_segment = height - thread_index * height_segment;

//	float* alpha_cache_start = m_alpha_cache[thread_index];
	// cache line structure: 
	// 4 floats of alpha_f from line 1
	// 4 floats of alpha_f from line 2
	// 4 floats of source color premultiplied with 'm_inv_alpha_f' from line 1
	// 4 floats of source color premultiplied with 'm_inv_alpha_f' from line 2
	// 4 floats of 1st pass result color from line 1
	// 4 floats of 1st pass result color from line 2
	float* line_cache = m_h_line_cache[thread_index];
	const float* range_table = m_range_table;

	__declspec(align(32)) long color_diff[16];

	_mm256_zeroall();

	__m256i mask_unpack = _mm256_setr_epi8(12, -1, -1, -1,		// pixel 1 R
		13, -1, -1, -1,		// pixel 1 G
		14, -1, -1, -1,		// pixel 1 B
		15, -1, -1, -1,		// pixel 1 A
		12, -1, -1, -1, // pixel 2 R
		13, -1, -1, -1, // pixel 2 G
		14, -1, -1, -1, // pixel 2 B
		15, -1, -1, -1);// pixel 2 A

	__m256i mask_pack = _mm256_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 4, 8, 12, // pixel 1
		-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 4, 8, 12); // pixel 2

	__m256 inv_alpha = _mm256_set1_ps(m_inv_alpha_f);

	// process 2 horizontal lines at a time
	for (int y = 0; y < height_segment; y+= 2)
	{
		__m256 alpha_prev = _mm256_set1_ps(1.f);
		__m256 color_prev;

		
		float* line_buffer = line_cache + 24 * pitch / 4;
		// 1st line
		int buffer_inc = (y + 1) * pitch - 32;
		const __m256i* src1_8xCur = (const __m256i*)(img_src + buffer_inc);
		const __m256i* src1_8xPrev = (const __m256i*)(img_src + buffer_inc + 4);
		// 2nd line
		buffer_inc += pitch;
		const __m256i* src2_8xCur = (const __m256i*)(img_src + buffer_inc);
		const __m256i* src2_8xPrev = (const __m256i*)(img_src + buffer_inc + 4);
		

		/////////////////////////////
		// right to left pass
		for (int x = 0; x < width32; x++)
		{
			__m256i pix8_1 = _mm256_load_si256(src1_8xCur--);
			__m256i pix8p_1 = _mm256_loadu_si256(src1_8xPrev--);
			getDiffFactor3x(pix8_1, pix8p_1, (__m256i*)color_diff);

			__m256i pix8_2 = _mm256_load_si256(src2_8xCur--);
			__m256i pix8p_2 = _mm256_loadu_si256(src2_8xPrev--);
			getDiffFactor3x(pix8_2, pix8p_2, (__m256i*)(color_diff + 8));

			// last 4 pixels of 2 lines
			__m256i pix8 = _mm256_permute2f128_si256(pix8_1, pix8_2, 1 | (3 << 4));

			////////////////////
			// pixel 1 unpack
			{
				// alpha factor
				float alpha2_f = range_table[color_diff[7]];
				float alpha1_f = range_table[color_diff[7 + 8]];
				__m256 alpha_f_8x = _mm256_set_ps(alpha1_f, alpha1_f, alpha1_f, alpha1_f,
					alpha2_f, alpha2_f, alpha2_f, alpha2_f);
				_mm256_store_ps(line_buffer, alpha_f_8x); // cache weights

				// source pixel
				__m256i pix2i = _mm256_shuffle_epi8(pix8, mask_unpack); // extracts 1 pixel components from BYTE to DWORD
				__m256 pix2f = _mm256_cvtepi32_ps(pix2i); // convert to floats
				if (x == 0) // have to initialize prev_color with last pixel color, this condition has no noticeable penalty
					color_prev = pix2f;
				pix2f = _mm256_mul_ps(pix2f, inv_alpha); // pre-multiply source color
				_mm256_store_ps(line_buffer + 8, pix2f);

				// filter 
				alpha_prev = _mm256_fmadd_ps(alpha_prev, alpha_f_8x, inv_alpha); // filter factor
				color_prev = _mm256_fmadd_ps(color_prev, alpha_f_8x, pix2f); // filter color

				// final color
				__m256 out_color = _mm256_div_ps(color_prev, alpha_prev); // get final color
				_mm256_store_ps(line_buffer + 16, out_color); // cache final color
				line_buffer -= 24;
			}

			////////////////////
			// pixel 2 unpack
			{
				// alpha factor
				float alpha2_f = range_table[color_diff[6]];
				float alpha1_f = range_table[color_diff[6 + 8]];
				__m256 alpha_f_8x = _mm256_set_ps(alpha1_f, alpha1_f, alpha1_f, alpha1_f,
					alpha2_f, alpha2_f, alpha2_f, alpha2_f);
				_mm256_store_ps(line_buffer, alpha_f_8x); // cache weights

				// source pixel
				pix8 = _mm256_slli_si256(pix8, 4); // shift left to unpack next pixel
				__m256i pix2i = _mm256_shuffle_epi8(pix8, mask_unpack); // extracts 1 pixel components from BYTE to DWORD
				__m256 pix2f = _mm256_cvtepi32_ps(pix2i); // convert to floats
				pix2f = _mm256_mul_ps(pix2f, inv_alpha); // pre-multiply source color
				_mm256_store_ps(line_buffer + 8, pix2f);

				// filter 
				alpha_prev = _mm256_fmadd_ps(alpha_prev, alpha_f_8x, inv_alpha); // filter factor
				color_prev = _mm256_fmadd_ps(color_prev, alpha_f_8x, pix2f); // filter color

				// final color
				__m256 out_color = _mm256_div_ps(color_prev, alpha_prev); // get final color
				_mm256_store_ps(line_buffer + 16, out_color); // cache final color
				line_buffer -= 24;
			}


			////////////////////
			// pixel 3 unpack
			{
				// alpha factor
				float alpha2_f = range_table[color_diff[5]];
				float alpha1_f = range_table[color_diff[5 + 8]];
				__m256 alpha_f_8x = _mm256_set_ps(alpha1_f, alpha1_f, alpha1_f, alpha1_f,
					alpha2_f, alpha2_f, alpha2_f, alpha2_f);
				_mm256_store_ps(line_buffer, alpha_f_8x); // cache weights

				// source pixel
				pix8 = _mm256_slli_si256(pix8, 4); // shift left to unpack next pixel
				__m256i pix2i = _mm256_shuffle_epi8(pix8, mask_unpack); // extracts 1 pixel components from BYTE to DWORD
				__m256 pix2f = _mm256_cvtepi32_ps(pix2i); // convert to floats
				pix2f = _mm256_mul_ps(pix2f, inv_alpha); // pre-multiply source color
				_mm256_store_ps(line_buffer + 8, pix2f);

				// filter 
				alpha_prev = _mm256_fmadd_ps(alpha_prev, alpha_f_8x, inv_alpha); // filter factor
				color_prev = _mm256_fmadd_ps(color_prev, alpha_f_8x, pix2f); // filter color

				// final color
				__m256 out_color = _mm256_div_ps(color_prev, alpha_prev); // get final color
				_mm256_store_ps(line_buffer + 16, out_color); // cache final color
				line_buffer -= 24;
			}

			////////////////////
			// pixel 4 unpack
			{
				// alpha factor
				float alpha2_f = range_table[color_diff[4]];
				float alpha1_f = range_table[color_diff[4 + 8]];
				__m256 alpha_f_8x = _mm256_set_ps(alpha1_f, alpha1_f, alpha1_f, alpha1_f,
					alpha2_f, alpha2_f, alpha2_f, alpha2_f);
				_mm256_store_ps(line_buffer, alpha_f_8x); // cache weights

				// source pixel
				pix8 = _mm256_slli_si256(pix8, 4); // shift left to unpack next pixel
				__m256i pix2i = _mm256_shuffle_epi8(pix8, mask_unpack); // extracts 1 pixel components from BYTE to DWORD
				__m256 pix2f = _mm256_cvtepi32_ps(pix2i); // convert to floats
				pix2f = _mm256_mul_ps(pix2f, inv_alpha); // pre-multiply source color
				_mm256_store_ps(line_buffer + 8, pix2f);

				// filter 
				alpha_prev = _mm256_fmadd_ps(alpha_prev, alpha_f_8x, inv_alpha); // filter factor
				color_prev = _mm256_fmadd_ps(color_prev, alpha_f_8x, pix2f); // filter color

				// final color
				__m256 out_color = _mm256_div_ps(color_prev, alpha_prev); // get final color
				_mm256_store_ps(line_buffer + 16, out_color); // cache final color
				line_buffer -= 24;
			}

			// next 4 pixels of 2 lines
			pix8 = _mm256_permute2f128_si256(pix8_1, pix8_2, 2 << 4);


			////////////////////
			// pixel 5 unpack
			{
				// alpha factor
				float alpha2_f = range_table[color_diff[3]];
				float alpha1_f = range_table[color_diff[3 + 8]];
				__m256 alpha_f_8x = _mm256_set_ps(alpha1_f, alpha1_f, alpha1_f, alpha1_f,
					alpha2_f, alpha2_f, alpha2_f, alpha2_f);
				_mm256_store_ps(line_buffer, alpha_f_8x); // cache weights

				// source pixel
				__m256i pix2i = _mm256_shuffle_epi8(pix8, mask_unpack); // extracts 1 pixel components from BYTE to DWORD
				__m256 pix2f = _mm256_cvtepi32_ps(pix2i); // convert to floats
				pix2f = _mm256_mul_ps(pix2f, inv_alpha); // pre-multiply source color
				_mm256_store_ps(line_buffer + 8, pix2f);

				// filter 
				alpha_prev = _mm256_fmadd_ps(alpha_prev, alpha_f_8x, inv_alpha); // filter factor
				color_prev = _mm256_fmadd_ps(color_prev, alpha_f_8x, pix2f); // filter color

																			 // final color
				__m256 out_color = _mm256_div_ps(color_prev, alpha_prev); // get final color
				_mm256_store_ps(line_buffer + 16, out_color); // cache final color
				line_buffer -= 24;
			}


			////////////////////
			// pixel 6 unpack
			{
				// alpha factor
				float alpha2_f = range_table[color_diff[2]];
				float alpha1_f = range_table[color_diff[2 + 8]];
				__m256 alpha_f_8x = _mm256_set_ps(alpha1_f, alpha1_f, alpha1_f, alpha1_f,
					alpha2_f, alpha2_f, alpha2_f, alpha2_f);
				_mm256_store_ps(line_buffer, alpha_f_8x); // cache weights

				// source pixel
				pix8 = _mm256_slli_si256(pix8, 4); // shift left to unpack next pixel
				__m256i pix2i = _mm256_shuffle_epi8(pix8, mask_unpack); // extracts 1 pixel components from BYTE to DWORD
				__m256 pix2f = _mm256_cvtepi32_ps(pix2i); // convert to floats
				pix2f = _mm256_mul_ps(pix2f, inv_alpha); // pre-multiply source color
				_mm256_store_ps(line_buffer + 8, pix2f);

				// filter 
				alpha_prev = _mm256_fmadd_ps(alpha_prev, alpha_f_8x, inv_alpha); // filter factor
				color_prev = _mm256_fmadd_ps(color_prev, alpha_f_8x, pix2f); // filter color

				// final color
				__m256 out_color = _mm256_div_ps(color_prev, alpha_prev); // get final color
				_mm256_store_ps(line_buffer + 16, out_color); // cache final color
				line_buffer -= 24;
			}


			////////////////////
			// pixel 7 unpack
			{
				// alpha factor
				float alpha2_f = range_table[color_diff[1]];
				float alpha1_f = range_table[color_diff[1 + 8]];
				__m256 alpha_f_8x = _mm256_set_ps(alpha1_f, alpha1_f, alpha1_f, alpha1_f,
					alpha2_f, alpha2_f, alpha2_f, alpha2_f);
				_mm256_store_ps(line_buffer, alpha_f_8x); // cache weights

				// source pixel
				pix8 = _mm256_slli_si256(pix8, 4); // shift left to unpack next pixel
				__m256i pix2i = _mm256_shuffle_epi8(pix8, mask_unpack); // extracts 1 pixel components from BYTE to DWORD
				__m256 pix2f = _mm256_cvtepi32_ps(pix2i); // convert to floats
				pix2f = _mm256_mul_ps(pix2f, inv_alpha); // pre-multiply source color
				_mm256_store_ps(line_buffer + 8, pix2f);

				// filter 
				alpha_prev = _mm256_fmadd_ps(alpha_prev, alpha_f_8x, inv_alpha); // filter factor
				color_prev = _mm256_fmadd_ps(color_prev, alpha_f_8x, pix2f); // filter color

				// final color
				__m256 out_color = _mm256_div_ps(color_prev, alpha_prev); // get final color
				_mm256_store_ps(line_buffer + 16, out_color); // cache final color
				line_buffer -= 24;
			}


			////////////////////
			// pixel 8 unpack
			{
				// alpha factor
				float alpha2_f = range_table[color_diff[0]];
				float alpha1_f = range_table[color_diff[0 + 8]];
				__m256 alpha_f_8x = _mm256_set_ps(alpha1_f, alpha1_f, alpha1_f, alpha1_f,
					alpha2_f, alpha2_f, alpha2_f, alpha2_f);
				_mm256_store_ps(line_buffer, alpha_f_8x); // cache weights

				// source pixel
				pix8 = _mm256_slli_si256(pix8, 4); // shift left to unpack next pixel
				__m256i pix2i = _mm256_shuffle_epi8(pix8, mask_unpack); // extracts 1 pixel components from BYTE to DWORD
				__m256 pix2f = _mm256_cvtepi32_ps(pix2i); // convert to floats
				pix2f = _mm256_mul_ps(pix2f, inv_alpha); // pre-multiply source color
				_mm256_store_ps(line_buffer + 8, pix2f);

				// filter 
				alpha_prev = _mm256_fmadd_ps(alpha_prev, alpha_f_8x, inv_alpha); // filter factor
				color_prev = _mm256_fmadd_ps(color_prev, alpha_f_8x, pix2f); // filter color

				// final color
				__m256 out_color = _mm256_div_ps(color_prev, alpha_prev); // get final color
				_mm256_store_ps(line_buffer + 16, out_color); // cache final color
				line_buffer -= 24;
			}

			
		}

		/////////////////////////////
		// left to right pass
		__m256i* dst1_pix8 = (__m256i*)(img_dst + y * pitch);
		__m256i* dst2_pix8 = (__m256i*)(img_dst + (y + 1) * pitch);

		for (int x = 0; x < width32; x++)
		{
			__m256i result1;
			__m256i result2;

			/////////////
			// 1st 4 pixels
			// pixel 1
			{
				// alpha
				__m256 alpha_f_8x = _mm256_load_ps(line_buffer);
				line_buffer += 24;

				// get pre-multiplied source color
				__m256 pix2f = _mm256_load_ps(line_buffer + 8);

				// first pixel in line needs to initialize color_prev to original source color
				if (x == 0)
					color_prev = _mm256_div_ps(pix2f, inv_alpha); // source color was premultiplied

				// filter 
				alpha_prev = _mm256_fmadd_ps(alpha_prev, alpha_f_8x, inv_alpha); // filter factor
				color_prev = _mm256_fmadd_ps(color_prev, alpha_f_8x, pix2f); // filter color

				// final color 
				__m256 out_color = _mm256_div_ps(color_prev, alpha_prev); // get final color

				// get final color from previous pass
				__m256 pix2f_p = _mm256_load_ps(line_buffer + 16);
				out_color = _mm256_add_ps(out_color, pix2f_p); // combine it with current final color
				__m256i pix2i = _mm256_cvtps_epi32(out_color); // covert to integer
				pix2i = _mm256_srli_epi32(pix2i, 1); // division by 2

				// pack result
				result1 = _mm256_shuffle_epi8(pix2i, mask_pack);
			}
			

			// pixel 2
			{
				// alpha
				__m256 alpha_f_8x = _mm256_load_ps(line_buffer);
				line_buffer += 24;

				// get pre-multiplied source color
				__m256 pix2f = _mm256_load_ps(line_buffer + 8);

				// filter 
				alpha_prev = _mm256_fmadd_ps(alpha_prev, alpha_f_8x, inv_alpha); // filter factor
				color_prev = _mm256_fmadd_ps(color_prev, alpha_f_8x, pix2f); // filter color

				// final color
				__m256 out_color = _mm256_div_ps(color_prev, alpha_prev); // get final color

				// get final color from previous pass
				__m256 pix2f_p = _mm256_load_ps(line_buffer + 16);
				out_color = _mm256_add_ps(out_color, pix2f_p); // combine it with current final color
				__m256i pix2i = _mm256_cvtps_epi32(out_color); // covert to integer
				pix2i = _mm256_srli_epi32(pix2i, 1); // division by 2

				// pack result
				pix2i = _mm256_shuffle_epi8(pix2i, mask_pack);
				result1 = _mm256_srli_si256(result1, 4); // shift 
				result1 = _mm256_or_si256(result1, pix2i); // combine
			}

			// pixel 3
			{
				// alpha
				__m256 alpha_f_8x = _mm256_load_ps(line_buffer);
				line_buffer += 24;

				// get pre-multiplied source color
				__m256 pix2f = _mm256_load_ps(line_buffer + 8);

				// filter 
				alpha_prev = _mm256_fmadd_ps(alpha_prev, alpha_f_8x, inv_alpha); // filter factor
				color_prev = _mm256_fmadd_ps(color_prev, alpha_f_8x, pix2f); // filter color

				// final color
				__m256 out_color = _mm256_div_ps(color_prev, alpha_prev); // get final color

				// get final color from previous pass
				__m256 pix2f_p = _mm256_load_ps(line_buffer + 16);
				out_color = _mm256_add_ps(out_color, pix2f_p); // combine it with current final color
				__m256i pix2i = _mm256_cvtps_epi32(out_color); // covert to integer
				pix2i = _mm256_srli_epi32(pix2i, 1); // division by 2

				// pack result
				pix2i = _mm256_shuffle_epi8(pix2i, mask_pack);
				result1 = _mm256_srli_si256(result1, 4); // shift 
				result1 = _mm256_or_si256(result1, pix2i); // combine
			}

			// pixel 4
			{
				// alpha
				__m256 alpha_f_8x = _mm256_load_ps(line_buffer);
				line_buffer += 24;

				// get pre-multiplied source color
				__m256 pix2f = _mm256_load_ps(line_buffer + 8);

				// filter 
				alpha_prev = _mm256_fmadd_ps(alpha_prev, alpha_f_8x, inv_alpha); // filter factor
				color_prev = _mm256_fmadd_ps(color_prev, alpha_f_8x, pix2f); // filter color

				// final color
				__m256 out_color = _mm256_div_ps(color_prev, alpha_prev); // get final color

				// get final color from previous pass
				__m256 pix2f_p = _mm256_load_ps(line_buffer + 16);
				out_color = _mm256_add_ps(out_color, pix2f_p); // combine it with current final color
				__m256i pix2i = _mm256_cvtps_epi32(out_color); // covert to integer
				pix2i = _mm256_srli_epi32(pix2i, 1); // division by 2

				// pack result
				pix2i = _mm256_shuffle_epi8(pix2i, mask_pack);
				result1 = _mm256_srli_si256(result1, 4); // shift 
				result1 = _mm256_or_si256(result1, pix2i); // combine
			}
		
			// next 4 pixels packed in result2
			// pixel 5	
			{
				// alpha
				__m256 alpha_f_8x = _mm256_load_ps(line_buffer);
				line_buffer += 24;

				// get pre-multiplied source color
				__m256 pix2f = _mm256_load_ps(line_buffer + 8);

				// filter 
				alpha_prev = _mm256_fmadd_ps(alpha_prev, alpha_f_8x, inv_alpha); // filter factor
				color_prev = _mm256_fmadd_ps(color_prev, alpha_f_8x, pix2f); // filter color

				// final color
				__m256 out_color = _mm256_div_ps(color_prev, alpha_prev); // get final color

				// get final color from previous pass
				__m256 pix2f_p = _mm256_load_ps(line_buffer + 16);
				out_color = _mm256_add_ps(out_color, pix2f_p); // combine it with current final color
				__m256i pix2i = _mm256_cvtps_epi32(out_color); // covert to integer
				pix2i = _mm256_srli_epi32(pix2i, 1); // division by 2

				// pack result
				result2 = _mm256_shuffle_epi8(pix2i, mask_pack);
			}
			
			// pixel 6
			{
				// alpha
				__m256 alpha_f_8x = _mm256_load_ps(line_buffer);
				line_buffer += 24;

				// get pre-multiplied source color
				__m256 pix2f = _mm256_load_ps(line_buffer + 8);

				// filter 
				alpha_prev = _mm256_fmadd_ps(alpha_prev, alpha_f_8x, inv_alpha); // filter factor
				color_prev = _mm256_fmadd_ps(color_prev, alpha_f_8x, pix2f); // filter color

				// final color
				__m256 out_color = _mm256_div_ps(color_prev, alpha_prev); // get final color

				// get final color from previous pass
				__m256 pix2f_p = _mm256_load_ps(line_buffer + 16);
				out_color = _mm256_add_ps(out_color, pix2f_p); // combine it with current final color
				__m256i pix2i = _mm256_cvtps_epi32(out_color); // covert to integer
				pix2i = _mm256_srli_epi32(pix2i, 1); // division by 2

				// pack result
				pix2i = _mm256_shuffle_epi8(pix2i, mask_pack);
				result2 = _mm256_srli_si256(result2, 4); // shift 
				result2 = _mm256_or_si256(result2, pix2i); // combine
			}

			// pixel 7
			{
				// alpha
				__m256 alpha_f_8x = _mm256_load_ps(line_buffer);
				line_buffer += 24;

				// get pre-multiplied source color
				__m256 pix2f = _mm256_load_ps(line_buffer + 8);

				// filter 
				alpha_prev = _mm256_fmadd_ps(alpha_prev, alpha_f_8x, inv_alpha); // filter factor
				color_prev = _mm256_fmadd_ps(color_prev, alpha_f_8x, pix2f); // filter color

				// final color
				__m256 out_color = _mm256_div_ps(color_prev, alpha_prev); // get final color

				// get final color from previous pass
				__m256 pix2f_p = _mm256_load_ps(line_buffer + 16);
				out_color = _mm256_add_ps(out_color, pix2f_p); // combine it with current final color
				__m256i pix2i = _mm256_cvtps_epi32(out_color); // covert to integer
				pix2i = _mm256_srli_epi32(pix2i, 1); // division by 2

				// pack result
				pix2i = _mm256_shuffle_epi8(pix2i, mask_pack);
				result2 = _mm256_srli_si256(result2, 4); // shift 
				result2 = _mm256_or_si256(result2, pix2i); // combine
			}

			// pixel 8
			{
				// alpha
				__m256 alpha_f_8x = _mm256_load_ps(line_buffer);
				line_buffer += 24;

				// get pre-multiplied source color
				__m256 pix2f = _mm256_load_ps(line_buffer + 8);

				// filter 
				alpha_prev = _mm256_fmadd_ps(alpha_prev, alpha_f_8x, inv_alpha); // filter factor
				color_prev = _mm256_fmadd_ps(color_prev, alpha_f_8x, pix2f); // filter color

				// final color
				__m256 out_color = _mm256_div_ps(color_prev, alpha_prev); // get final color

				// get final color from previous pass
				__m256 pix2f_p = _mm256_load_ps(line_buffer + 16);
				out_color = _mm256_add_ps(out_color, pix2f_p); // combine it with current final color
				__m256i pix2i = _mm256_cvtps_epi32(out_color); // covert to integer
				pix2i = _mm256_srli_epi32(pix2i, 1); // division by 2

				// pack result
				pix2i = _mm256_shuffle_epi8(pix2i, mask_pack);
				result2 = _mm256_srli_si256(result2, 4); // shift 
				result2 = _mm256_or_si256(result2, pix2i); // combine
			}

			// separate packed results into lines
			__m256i line1 = _mm256_permute2f128_si256(result1, result2, 2 << 4);
			__m256i line2 = _mm256_permute2f128_si256(result1, result2, 1 | (3 << 4));

			// store result
			_mm256_store_si256(dst1_pix8++, line1);
			_mm256_store_si256(dst2_pix8++, line2);
		}
	}

}


void CRBFilterAVX2::verticalFilter(int thread_index, const unsigned char* img_src, unsigned char* img_dst, int width, int height, int pitch)
{
	int width_segment = width / m_thread_count;
	// make sure width segments round to 32 byte boundary
	width_segment -= width_segment % 8;
	int start_offset = width_segment * thread_index;
	if (thread_index == m_thread_count - 1) // last one
	{
		width_segment = getOptimalPitch(width) / 4 - start_offset;
	}

	int width8 = width_segment / 8;

	// adjust img buffer starting positions
	img_src += start_offset * 4;
	img_dst += start_offset * 4;

	float* line_cache = m_v_line_cache[thread_index];
	const float* range_table = m_range_table;

	_mm256_zeroall();

	__m256 inv_alpha = _mm256_set1_ps(m_inv_alpha_f);

	__m256i mask_pack = _mm256_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 4, 8, 12, // pixel 1
		-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 4, 8, 12); // pixel 2

	__m256i mask_unpack = _mm256_setr_epi8(0, -1, -1, -1, 1, -1, -1, -1, 2, -1, -1, -1, 3, -1, -1, -1, // pixel 1
										0, -1, -1, -1, 1, -1, -1, -1, 2, -1, -1, -1, 3, -1, -1, -1); // pixel 2

	// used to store maximum difference between 2 pixels
	__declspec(align(32)) long color_diff[8];

	/////////////////
	// Bottom to top pass first
	{
		// last line processed separately since no previous
		{
			float* line_buffer = line_cache;
			__m256i* dst_buf = (__m256i*)(img_dst + (height - 1) * pitch);
			__m256i* src_8xCur = (__m256i*)(img_src + (height - 1) * pitch);

			__m256 one = _mm256_set1_ps(1.f);

			for (int x = 0; x < width8; x++)
			{
				__m256i pix8 = _mm256_load_si256(src_8xCur++); // load 8x pixel
				_mm256_store_si256(dst_buf++, pix8); // copy to destination

				for (int i = 0; i < 4; i++)
				{
					__m256i pix2i = _mm256_shuffle_epi8(pix8, mask_unpack);
					pix8 = _mm256_srli_si256(pix8, 4); // shift right
					__m256 pix2f = _mm256_cvtepi32_ps(pix2i); // convert to floats

					_mm256_store_ps(line_buffer, one);
					_mm256_store_ps(line_buffer + 8, pix2f);

					line_buffer += 16;
				}
			}
		}

		// process other lines
		for (int y = height - 2; y >= 0; y--)
		{
			float* line_buffer = line_cache;
			__m256i* dst_buf = (__m256i*)(img_dst + y * pitch);
			__m256i* src_8xCur = (__m256i*)(img_src + y * pitch);
			__m256i* src_8xPrev = (__m256i*)(img_src + (y + 1) * pitch);

			for (int x = 0; x < width8; x++)
			{
				__m256i pix8 = _mm256_load_si256(src_8xCur++); // load 8x pixel
				__m256i pix8p = _mm256_load_si256(src_8xPrev++);
				__m256i pix_out; // final 8x packed pixels

				// get color differences
				getDiffFactor3x(pix8, pix8p, (__m256i*)color_diff);

				////////////////////
				// pixel 1, 5 unpack
				{
					// alpha factor
					float alpha2_f = range_table[color_diff[0]];
					float alpha1_f = range_table[color_diff[4]];
					__m256 alpha_f_8x = _mm256_set_ps(alpha1_f, alpha1_f, alpha1_f, alpha1_f,
						alpha2_f, alpha2_f, alpha2_f, alpha2_f);
					
					// load previous line color factor
					__m256 alpha_prev = _mm256_load_ps(line_buffer);
					// load previous line color
					__m256 color_prev = _mm256_load_ps(line_buffer + 8);
					
					// unpack current source pixel
					__m256i pix2i = _mm256_shuffle_epi8(pix8, mask_unpack);
					__m256 pix2f = _mm256_cvtepi32_ps(pix2i); // convert to floats

					// filter
					pix2f = _mm256_mul_ps(pix2f, inv_alpha);
					alpha_prev = _mm256_fmadd_ps(alpha_prev, alpha_f_8x, inv_alpha); // filter factor
					color_prev = _mm256_fmadd_ps(color_prev, alpha_f_8x, pix2f); // filter color
					
					// store current factor and color as previous for next cycle
					_mm256_store_ps(line_buffer, alpha_prev);
					_mm256_store_ps(line_buffer + 8, color_prev);
					line_buffer += 16;

					// calculate final color
					pix2f = _mm256_div_ps(color_prev, alpha_prev);

					// pack float pixel into byte pixel
					pix2i = _mm256_cvtps_epi32(pix2f); // convert to integer
					pix_out = _mm256_shuffle_epi8(pix2i, mask_pack);
				}
				
				// loop for other pixels
				for(int i=1; i<4; i++)
				{
					// alpha factor
					float alpha2_f = range_table[color_diff[i]];
					float alpha1_f = range_table[color_diff[i+4]];
					__m256 alpha_f_8x = _mm256_set_ps(alpha1_f, alpha1_f, alpha1_f, alpha1_f,
						alpha2_f, alpha2_f, alpha2_f, alpha2_f);

					// load previous line color factor
					__m256 alpha_prev = _mm256_load_ps(line_buffer);
					// load previous line color
					__m256 color_prev = _mm256_load_ps(line_buffer + 8);

					// unpack current source pixel
					pix8 = _mm256_srli_si256(pix8, 4); // shift right
					__m256i pix2i = _mm256_shuffle_epi8(pix8, mask_unpack);
					__m256 pix2f = _mm256_cvtepi32_ps(pix2i); // convert to floats

					// filter
					pix2f = _mm256_mul_ps(pix2f, inv_alpha);
					alpha_prev = _mm256_fmadd_ps(alpha_prev, alpha_f_8x, inv_alpha); // filter factor
					color_prev = _mm256_fmadd_ps(color_prev, alpha_f_8x, pix2f); // filter color

					// store current factor and color as previous for next cycle
					_mm256_store_ps(line_buffer, alpha_prev);
					_mm256_store_ps(line_buffer + 8, color_prev);
					line_buffer += 16;

					// calculate final color
					pix2f = _mm256_div_ps(color_prev, alpha_prev);

					// pack float pixel into byte pixel
					pix2i = _mm256_cvtps_epi32(pix2f); // convert to integer
					pix2i = _mm256_shuffle_epi8(pix2i, mask_pack);
					pix_out = _mm256_srli_si256(pix_out, 4); // shift 
					pix_out = _mm256_or_si256(pix_out, pix2i); // combine
				}
				
				// store result
				_mm256_store_si256(dst_buf++, pix_out);
			}
		}
	}

	/////////////////
	// Top to bottom pass last
	{

		// first line processed separately since no previous
		{
			float* line_buffer = line_cache;
			__m256i* dst_line = (__m256i*)img_dst;
			__m256i* src_8xCur = (__m256i*)img_src;

			__m256 one = _mm256_set1_ps(1.f);

			for (int x = 0; x < width8; x++)
			{
				__m256i pix8 = _mm256_load_si256(src_8xCur++); // load 8x pixel
				__m256i pix8_d = _mm256_load_si256(dst_line);
				pix8_d = _mm256_avg_epu8(pix8_d, pix8); // average out
				_mm256_store_si256(dst_line++, pix8_d);

				for (int i = 0; i < 4; i++)
				{
					__m256i pix2i = _mm256_shuffle_epi8(pix8, mask_unpack);
					pix8 = _mm256_srli_si256(pix8, 4); // shift right
					__m256 pix2f = _mm256_cvtepi32_ps(pix2i); // convert to floats

					_mm256_store_ps(line_buffer, one);
					_mm256_store_ps(line_buffer + 8, pix2f);

					line_buffer += 16;
				}
			}
		}

		// process other lines
		for (int y = 1; y < height; y++)
		{
			float* line_buffer = line_cache;
			__m256i* dst_buf = (__m256i*)(img_dst + y * pitch);
			__m256i* src_8xCur = (__m256i*)(img_src + y * pitch);
			__m256i* src_8xPrev = (__m256i*)(img_src + (y - 1) * pitch);

			for (int x = 0; x < width8; x++)
			{
				__m256i pix8 = _mm256_load_si256(src_8xCur++); // load 8x pixel
				__m256i pix8p = _mm256_load_si256(src_8xPrev++);
				__m256i pix_out; // final 8x packed pixels

				// get color differences
				getDiffFactor3x(pix8, pix8p, (__m256i*)color_diff);

				////////////////////
				// pixel 1, 5 unpack
				{
					// alpha factor
					float alpha2_f = range_table[color_diff[0]];
					float alpha1_f = range_table[color_diff[4]];
					__m256 alpha_f_8x = _mm256_set_ps(alpha1_f, alpha1_f, alpha1_f, alpha1_f,
						alpha2_f, alpha2_f, alpha2_f, alpha2_f);

					// load previous line color factor
					__m256 alpha_prev = _mm256_load_ps(line_buffer);
					// load previous line color
					__m256 color_prev = _mm256_load_ps(line_buffer + 8);

					// unpack current source pixel
					__m256i pix2i = _mm256_shuffle_epi8(pix8, mask_unpack);
					__m256 pix2f = _mm256_cvtepi32_ps(pix2i); // convert to floats

					// filter
					pix2f = _mm256_mul_ps(pix2f, inv_alpha);
					alpha_prev = _mm256_fmadd_ps(alpha_prev, alpha_f_8x, inv_alpha); // filter factor
					color_prev = _mm256_fmadd_ps(color_prev, alpha_f_8x, pix2f); // filter color

					// store current factor and color as previous for next cycle
					_mm256_store_ps(line_buffer, alpha_prev);
					_mm256_store_ps(line_buffer + 8, color_prev);
					line_buffer += 16;

					// calculate final color
					pix2f = _mm256_div_ps(color_prev, alpha_prev);

					// pack float pixel into byte pixel
					pix2i = _mm256_cvtps_epi32(pix2f); // convert to integer
					pix_out = _mm256_shuffle_epi8(pix2i, mask_pack);
				}

				// loop for other pixels
				for (int i = 1; i<4; i++)
				{
					// alpha factor
					float alpha2_f = range_table[color_diff[i]];
					float alpha1_f = range_table[color_diff[i + 4]];
					__m256 alpha_f_8x = _mm256_set_ps(alpha1_f, alpha1_f, alpha1_f, alpha1_f,
						alpha2_f, alpha2_f, alpha2_f, alpha2_f);

					// load previous line color factor
					__m256 alpha_prev = _mm256_load_ps(line_buffer);
					// load previous line color
					__m256 color_prev = _mm256_load_ps(line_buffer + 8);

					// unpack current source pixel
					pix8 = _mm256_srli_si256(pix8, 4); // shift right
					__m256i pix2i = _mm256_shuffle_epi8(pix8, mask_unpack);
					__m256 pix2f = _mm256_cvtepi32_ps(pix2i); // convert to floats

					// filter
					pix2f = _mm256_mul_ps(pix2f, inv_alpha);
					alpha_prev = _mm256_fmadd_ps(alpha_prev, alpha_f_8x, inv_alpha); // filter factor
					color_prev = _mm256_fmadd_ps(color_prev, alpha_f_8x, pix2f); // filter color

					// store current factor and color as previous for next cycle
					_mm256_store_ps(line_buffer, alpha_prev);
					_mm256_store_ps(line_buffer + 8, color_prev);
					line_buffer += 16;

					// calculate final color
					pix2f = _mm256_div_ps(color_prev, alpha_prev);

					// pack float pixel into byte pixel
					pix2i = _mm256_cvtps_epi32(pix2f); // convert to integer
					pix2i = _mm256_shuffle_epi8(pix2i, mask_pack);
					pix_out = _mm256_srli_si256(pix_out, 4); // shift 
					pix_out = _mm256_or_si256(pix_out, pix2i); // combine
				}

				// average result with previous values in destination buffer
				__m256i pix8_d = _mm256_load_si256(dst_buf);
				pix_out = _mm256_avg_epu8(pix8_d, pix_out);
				_mm256_store_si256(dst_buf++, pix_out);
			}
		}
	}
}

bool CRBFilterAVX2::filter(unsigned char* out_data, const unsigned char* in_data, int width, int height, int pitch)
{
	// basic error checking
	if (!m_stage_buffer[0])
		return false;

	if (width < 32 || width > m_reserved_width)
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
		m_horizontal_tasks[i] = std::async(std::launch::async, &CRBFilterAVX2::horizontalFilter, this, i, in_data, m_stage_buffer[0], width, height, pitch);
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
		m_vertical_tasks[i] = std::async(std::launch::async, &CRBFilterAVX2::verticalFilter, this, i, m_stage_buffer[0], out_data, width, height, pitch);
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

bool CRBFilterAVX2::filterPipePush(unsigned char* out_data, const unsigned char* in_data, int width, int height, int pitch)
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
		if(m_horizontal_tasks[i].valid())
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
			m_horizontal_tasks[i] = std::async(std::launch::async, &CRBFilterAVX2::horizontalFilter, this, i, in_data, m_stage_buffer[current_stage_index], width, height, pitch);
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
			m_vertical_tasks[i] = std::async(std::launch::async, &CRBFilterAVX2::verticalFilter, this, i, m_stage_buffer[previous_stage_index], m_out_buffer[previous_stage_index], width, height, pitch);
		}
	}
	
	return true;
}

void CRBFilterAVX2::filterPipeFlush()
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