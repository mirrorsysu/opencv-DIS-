- processPatchMeanNorm
```cpp
/* Same as processPatch, but with patch mean normalization, which improves robustness under changing
 * lighting conditions
 */
inline float processPatchMeanNorm(float &dst_dUx, float &dst_dUy, uchar *I0_ptr, uchar *I1_ptr, short *I0x_ptr,
                                  short *I0y_ptr, int I0_stride, int I1_stride, float w00, float w01, float w10,
                                  float w11, int patch_sz, float x_grad_sum, float y_grad_sum)
{
    float sum_diff = 0.0, sum_diff_sq = 0.0;
    float sum_I0x_mul = 0.0, sum_I0y_mul = 0.0;
    float n = (float)patch_sz * patch_sz;

    // 删掉了simd的部分
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

                sum_I0x_mul += diff * I0x_ptr[i * I0_stride + j];
                sum_I0y_mul += diff * I0y_ptr[i * I0_stride + j];
            }
    }
    dst_dUx = sum_I0x_mul - sum_diff * x_grad_sum / n;
    dst_dUy = sum_I0y_mul - sum_diff * y_grad_sum / n;
    return sum_diff_sq - sum_diff * sum_diff / n;
}

```