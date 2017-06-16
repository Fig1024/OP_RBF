#pragma once


// This class is useful only for the sake of understanding the main principles of Recursive Bilateral Filter
// It is designed in non-optimal but easy to understand way. It also does not match 1:1 with original, 
// some creative liberties were taken with original idea.
// This class is not used in performance tests

class CRBFilterPlain
{
	int			m_reserve_width = 0;
	int			m_reserve_height = 0;
	int			m_reserve_channels = 0;

	float*		m_left_pass_color = nullptr;
	float*		m_left_pass_factor = nullptr;

	float*		m_right_pass_color = nullptr;
	float*		m_right_pass_factor = nullptr;

	float*		m_down_pass_color = nullptr;
	float*		m_down_pass_factor = nullptr;

	float*		m_up_pass_color = nullptr;
	float*		m_up_pass_factor = nullptr;

	int getDiffFactor(const unsigned char* color1, const unsigned char* color2) const;

public:

	CRBFilterPlain();
	~CRBFilterPlain();

	// assumes 3/4 channel images, 1 byte per channel
	void reserveMemory(int max_width, int max_height, int channels);
	void releaseMemory();

	// memory must be reserved before calling image filter
	// this implementation of filter uses plain C++, single threaded
	// channel count must be 3 or 4 (alpha not used)
	void filter(unsigned char* img_src, unsigned char* img_dst,
		float sigma_spatial, float sigma_range,
		int width, int height, int channel);
};