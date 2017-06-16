#include "stdafx.h"
#include "RBFilterPlain.h"
#include "stdafx.h"
#include "RBFilterPlain.h"
#include <algorithm>

using namespace std;

#define QX_DEF_CHAR_MAX 255


CRBFilterPlain::CRBFilterPlain()
{

}

CRBFilterPlain::~CRBFilterPlain()
{
	releaseMemory();
}

// assumes 3/4 channel images, 1 byte per channel
void CRBFilterPlain::reserveMemory(int max_width, int max_height, int channels)
{
	// basic sanity check
	_ASSERT(max_width >= 10 && max_width < 10000);
	_ASSERT(max_height >= 10 && max_height < 10000);
	_ASSERT(channels >= 1 && channels <= 4);

	releaseMemory();

	m_reserve_width = max_width;
	m_reserve_height = max_height;
	m_reserve_channels = channels;

	int width_height = m_reserve_width * m_reserve_height;
	int width_height_channel = width_height * m_reserve_channels;

	m_left_pass_color = new float[width_height_channel];
	m_left_pass_factor = new float[width_height];

	m_right_pass_color = new float[width_height_channel];
	m_right_pass_factor = new float[width_height];

	m_down_pass_color = new float[width_height_channel];
	m_down_pass_factor = new float[width_height];

	m_up_pass_color = new float[width_height_channel];
	m_up_pass_factor = new float[width_height];
}

void CRBFilterPlain::releaseMemory()
{
	m_reserve_width = 0;
	m_reserve_height = 0;
	m_reserve_channels = 0;

	if (m_left_pass_color)
	{
		delete[] m_left_pass_color;
		m_left_pass_color = nullptr;
	}

	if (m_left_pass_factor)
	{
		delete[] m_left_pass_factor;
		m_left_pass_factor = nullptr;
	}

	if (m_right_pass_color)
	{
		delete[] m_right_pass_color;
		m_right_pass_color = nullptr;
	}

	if (m_right_pass_factor)
	{
		delete[] m_right_pass_factor;
		m_right_pass_factor = nullptr;
	}

	if (m_down_pass_color)
	{
		delete[] m_down_pass_color;
		m_down_pass_color = nullptr;
	}

	if (m_down_pass_factor)
	{
		delete[] m_down_pass_factor;
		m_down_pass_factor = nullptr;
	}

	if (m_up_pass_color)
	{
		delete[] m_up_pass_color;
		m_up_pass_color = nullptr;
	}

	if (m_up_pass_factor)
	{
		delete[] m_up_pass_factor;
		m_up_pass_factor = nullptr;
	}
}

int CRBFilterPlain::getDiffFactor(const unsigned char* color1, const unsigned char* color2) const
{
	int final_diff;
	int component_diff[4];

	// find absolute difference between each component
	for (int i = 0; i < m_reserve_channels; i++)
	{
		component_diff[i] = abs(color1[i] - color2[i]);
	}

	// based on number of components, produce a single difference value in the 0-255 range
	switch (m_reserve_channels)
	{
	case 1:
		final_diff = component_diff[0];
		break;

	case 2:
		final_diff = ((component_diff[0] + component_diff[1]) >> 1);
		break;

	case 3:
		final_diff = ((component_diff[0] + component_diff[2]) >> 2) + (component_diff[1] >> 1);
		break;

	case 4:
		final_diff = ((component_diff[0] + component_diff[1] + component_diff[2] + component_diff[3]) >> 2);
		break;

	default:
		final_diff = 0;
	}

	_ASSERT(final_diff >= 0 && final_diff <= 255);

	return final_diff;
}

// memory must be reserved before calling image filter
// this implementation of filter uses plain C++, single threaded
// channel count must be 3 or 4 (alpha not used)
void CRBFilterPlain::filter(unsigned char* img_src, unsigned char* img_dst,
	float sigma_spatial, float sigma_range,
	int width, int height, int channel)
{
	_ASSERT(img_src);
	_ASSERT(img_dst);
	_ASSERT(m_reserve_channels == channel);
	_ASSERT(m_reserve_width >= width);
	_ASSERT(m_reserve_height >= height);

	// compute a lookup table
	float alpha_f = static_cast<float>(exp(-sqrt(2.0) / (sigma_spatial * 255)));
	float inv_alpha_f = 1.f - alpha_f;


	float range_table_f[QX_DEF_CHAR_MAX + 1];
	float inv_sigma_range = 1.0f / (sigma_range * QX_DEF_CHAR_MAX);
	{
		float ii = 0.f;
		for (int i = 0; i <= QX_DEF_CHAR_MAX; i++, ii -= 1.f)
		{
			range_table_f[i] = alpha_f * exp(ii * inv_sigma_range);
		}
	}

	///////////////
	// Left pass
	{
		const unsigned char* src_color = img_src;
		float* left_pass_color = m_left_pass_color;
		float* left_pass_factor = m_left_pass_factor;

		for (int y = 0; y < height; y++)
		{
			const unsigned char* src_prev = src_color;
			const float* prev_factor = left_pass_factor;
			const float* prev_color = left_pass_color;

			// process 1st pixel separately since it has no previous
			*left_pass_factor++ = 1.f;
			for (int c = 0; c < channel; c++)
			{
				*left_pass_color++ = *src_color++;
			}

			// handle other pixels
			for (int x = 1; x < width; x++)
			{
				// determine difference in pixel color between current and previous
				// calculation is different depending on number of channels
				int diff = getDiffFactor(src_color, src_prev);
				src_prev = src_color;

				float alpha_f = range_table_f[diff];

				*left_pass_factor++ = inv_alpha_f + alpha_f * (*prev_factor++);

				for (int c = 0; c < channel; c++)
				{
					*left_pass_color++ = inv_alpha_f * (*src_color++) + alpha_f * (*prev_color++);
				}
			}
		}
	}

	///////////////
	// Right pass
	{
		// start from end and then go up to begining 
		int last_index = width * height * channel - 1;
		const unsigned char* src_color = img_src + last_index;
		float* right_pass_color = m_right_pass_color + last_index;
		float* right_pass_factor = m_right_pass_factor + width * height - 1;

		for (int y = 0; y < height; y++)
		{
			const unsigned char* src_prev = src_color;
			const float* prev_factor = right_pass_factor;
			const float* prev_color = right_pass_color;

			// process 1st pixel separately since it has no previous
			*right_pass_factor-- = 1.f;
			for (int c = 0; c < channel; c++)
			{
				*right_pass_color-- = *src_color--;
			}

			// handle other pixels
			for (int x = 1; x < width; x++)
			{
				// determine difference in pixel color between current and previous
				// calculation is different depending on number of channels
				int diff = getDiffFactor(src_color, src_color - 3);
				//	src_prev = src_color;

				float alpha_f = range_table_f[diff];

				*right_pass_factor-- = inv_alpha_f + alpha_f * (*prev_factor--);

				for (int c = 0; c < channel; c++)
				{
					*right_pass_color-- = inv_alpha_f * (*src_color--) + alpha_f * (*prev_color--);
				}
			}
		}
	}

	// vertical pass will be applied on top on horizontal pass, while using pixel differences from original image
	// result color stored in 'm_left_pass_color' and vertical pass will use it as source color
	{
		float* img_out = m_left_pass_color; // use as temporary buffer
		const float* left_pass_color = m_left_pass_color;
		const float* left_pass_factor = m_left_pass_factor;
		const float* right_pass_color = m_right_pass_color;
		const float* right_pass_factor = m_right_pass_factor;

		int width_height = width * height;
		for (int i = 0; i < width_height; i++)
		{
			// average color divided by average factor
			float factor = 1.f / ((*left_pass_factor++) + (*right_pass_factor++));
			for (int c = 0; c < channel; c++)
			{
				*img_out++ = (factor * ((*left_pass_color++) + (*right_pass_color++)));
			}
		}
	}

	///////////////
	// Down pass
	{
		const float* src_color_hor = m_left_pass_color; // result of horizontal pass filter

		const unsigned char* src_color = img_src;
		float* down_pass_color = m_down_pass_color;
		float* down_pass_factor = m_down_pass_factor;

		const unsigned char* src_prev = src_color;
		const float* prev_color = down_pass_color;
		const float* prev_factor = down_pass_factor;

		// 1st line done separately because no previous line
		for (int x = 0; x < width; x++)
		{
			*down_pass_factor++ = 1.f;
			for (int c = 0; c < channel; c++)
			{
				*down_pass_color++ = *src_color_hor++;
			}
			src_color += channel;
		}

		// handle other lines
		for (int y = 1; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				// determine difference in pixel color between current and previous
				// calculation is different depending on number of channels
				int diff = getDiffFactor(src_color, src_prev);
				src_prev += channel;
				src_color += channel;

				float alpha_f = range_table_f[diff];

				*down_pass_factor++ = inv_alpha_f + alpha_f * (*prev_factor++);

				for (int c = 0; c < channel; c++)
				{
					*down_pass_color++ = inv_alpha_f * (*src_color_hor++) + alpha_f * (*prev_color++);
				}
			}
		}
	}

	///////////////
	// Up pass
	{
		// start from end and then go up to begining 
		int last_index = width * height * channel - 1;
		const unsigned char* src_color = img_src + last_index;
		const float* src_color_hor = m_left_pass_color + last_index; // result of horizontal pass filter
		float* up_pass_color = m_up_pass_color + last_index;
		float* up_pass_factor = m_up_pass_factor + (width * height - 1);

		//	const unsigned char* src_prev = src_color;
		const float* prev_color = up_pass_color;
		const float* prev_factor = up_pass_factor;

		// 1st line done separately because no previous line
		for (int x = 0; x < width; x++)
		{
			*up_pass_factor-- = 1.f;
			for (int c = 0; c < channel; c++)
			{
				*up_pass_color-- = *src_color_hor--;
			}
			src_color -= channel;
		}

		// handle other lines
		for (int y = 1; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				// determine difference in pixel color between current and previous
				// calculation is different depending on number of channels
				src_color -= channel;
				int diff = getDiffFactor(src_color, src_color + width * channel);

				float alpha_f = range_table_f[diff];

				*up_pass_factor-- = inv_alpha_f + alpha_f * (*prev_factor--);

				for (int c = 0; c < channel; c++)
				{
					*up_pass_color-- = inv_alpha_f * (*src_color_hor--) + alpha_f * (*prev_color--);
				}
			}
		}
	}

	///////////////
	// average result of vertical pass is written to output buffer
	{
		const float* down_pass_color = m_down_pass_color;
		const float* down_pass_factor = m_down_pass_factor;
		const float* up_pass_color = m_up_pass_color;
		const float* up_pass_factor = m_up_pass_factor;

		int width_height = width * height;
		for (int i = 0; i < width_height; i++)
		{
			// average color divided by average factor
			float factor = 1.f / ((*up_pass_factor++) + (*down_pass_factor++));
			for (int c = 0; c < channel; c++)
			{
				*img_dst++ = (unsigned char)(factor * ((*up_pass_color++) + (*down_pass_color++)));
			}
		}
	}
}
