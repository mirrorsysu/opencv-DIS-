- computeSSDMeanNorm
```cpp
/* Same as computeSSD, but with patch mean normalization */
inline float computeSSDMeanNorm(uchar *I0_ptr, uchar *I1_ptr, int I0_stride, int I1_stride, float w00, float w01,
                                float w10, float w11, int patch_sz)
{
    float sum_diff = 0.0f, sum_diff_sq = 0.0f;
    float n = (float)patch_sz * patch_sz;

    // 去掉了simd部分
    {
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
    }
    return sum_diff_sq - sum_diff * sum_diff / n;
}

```