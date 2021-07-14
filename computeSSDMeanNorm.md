- computeSSDMeanNorm
```cpp
/* Same as computeSSD, but with patch mean normalization */
inline float computeSSDMeanNorm(uchar *I0_ptr, uchar *I1_ptr, int I0_stride, int I1_stride, float w00, float w01,
                                float w10, float w11, int patch_sz)
{
    float sum_diff = 0.0f, sum_diff_sq = 0.0f;
    float n = (float)patch_sz * patch_sz;
#if CV_SIMD128
    if (patch_sz == 8)
    {
        v_float32x4 sum_diff_vec = v_setall_f32(0);
        v_float32x4 sum_diff_sq_vec = v_setall_f32(0);
        HAL_INIT_BILINEAR_8x8_PATCH_EXTRACTION;
        for (int row = 0; row < 8; row++)
        {
            HAL_PROCESS_BILINEAR_8x8_PATCH_EXTRACTION;
            sum_diff_sq_vec += I_diff_left * I_diff_left + I_diff_right * I_diff_right;
            sum_diff_vec += I_diff_left + I_diff_right;
            HAL_BILINEAR_8x8_PATCH_EXTRACTION_NEXT_ROW;
        }
        sum_diff = v_reduce_sum(sum_diff_vec);
        sum_diff_sq = v_reduce_sum(sum_diff_sq_vec);
    }
    else
    {
#endif
        float diff;
        for (int i = 0; i < patch_sz; i++)
            for (int j = 0; j < patch_sz; j++)
            {
                diff = w00 * I1_ptr[i * I1_stride + j] + w01 * I1_ptr[i * I1_stride + j + 1] +
                       w10 * I1_ptr[(i + 1) * I1_stride + j] + w11 * I1_ptr[(i + 1) * I1_stride + j + 1] -
                       I0_ptr[i * I0_stride + j];

                sum_diff += diff;
                sum_diff_sq += diff * diff;
            }
#if CV_SIMD128
    }
#endif
    return sum_diff_sq - sum_diff * sum_diff / n;
}

```