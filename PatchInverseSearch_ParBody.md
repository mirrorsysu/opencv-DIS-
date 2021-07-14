
- PatchInverseSearch_ParBody
```cpp
struct PatchInverseSearch_ParBody : public ParallelLoopBody
{
    DISOpticalFlowImpl *dis;
    int nstripes, stripe_sz;
    int hs;
    Mat *Sx, *Sy, *Ux, *Uy, *I0, *I1, *I0x, *I0y;
    int num_iter, pyr_level;

    PatchInverseSearch_ParBody(DISOpticalFlowImpl &_dis, int _nstripes, int _hs, Mat &dst_Sx, Mat &dst_Sy,
                                Mat &src_Ux, Mat &src_Uy, Mat &_I0, Mat &_I1, Mat &_I0x, Mat &_I0y, int _num_iter,
                                int _pyr_level);
    void operator()(const Range &range) const CV_OVERRIDE;
};

DISOpticalFlowImpl::PatchInverseSearch_ParBody::PatchInverseSearch_ParBody(DISOpticalFlowImpl &_dis, int _nstripes,
                                                                        int _hs, Mat &dst_Sx, Mat &dst_Sy,
                                                                        Mat &src_Ux, Mat &src_Uy, Mat &_I0, Mat &_I1,
                                                                        Mat &_I0x, Mat &_I0y, int _num_iter,
                                                                        int _pyr_level)
    : dis(&_dis), nstripes(_nstripes), hs(_hs), Sx(&dst_Sx), Sy(&dst_Sy), Ux(&src_Ux), Uy(&src_Uy), I0(&_I0), I1(&_I1),
        I0x(&_I0x), I0y(&_I0y), num_iter(_num_iter), pyr_level(_pyr_level)
{
    stripe_sz = (int)ceil(hs / (double)nstripes);
}

void DISOpticalFlowImpl::PatchInverseSearch_ParBody::operator()(const Range &range) const
{
    CV_INSTRUMENT_REGION();

    // force separate processing of stripes if we are using spatial propagation:
    if (dis->use_spatial_propagation && range.end > range.start + 1)
    {
        for (int n = range.start; n < range.end; n++)
            (*this)(Range(n, n + 1));
        return;
    }
    int psz = dis->patch_size;
    int psz2 = psz / 2;
    int w_ext = dis->w + 2 * dis->border_size; //!< width of I1_ext
    int bsz = dis->border_size;

    /* Input dense flow */
    float *Ux_ptr = Ux->ptr<float>();
    float *Uy_ptr = Uy->ptr<float>();

    /* Output sparse flow */
    float *Sx_ptr = Sx->ptr<float>();
    float *Sy_ptr = Sy->ptr<float>();

    uchar *I0_ptr = I0->ptr<uchar>();
    uchar *I1_ptr = I1->ptr<uchar>();
    short *I0x_ptr = I0x->ptr<short>();
    short *I0y_ptr = I0y->ptr<short>();

    /* Precomputed structure tensor */
    float *xx_ptr = dis->I0xx_buf.ptr<float>();
    float *yy_ptr = dis->I0yy_buf.ptr<float>();
    float *xy_ptr = dis->I0xy_buf.ptr<float>();
    /* And extra buffers for mean-normalization: */
    float *x_ptr = dis->I0x_buf.ptr<float>();
    float *y_ptr = dis->I0y_buf.ptr<float>();

    bool use_temporal_candidates = false;
    float *initial_Ux_ptr = NULL, *initial_Uy_ptr = NULL;
    if (!dis->initial_Ux.empty())
    {
        initial_Ux_ptr = dis->initial_Ux[pyr_level].ptr<float>();
        initial_Uy_ptr = dis->initial_Uy[pyr_level].ptr<float>();
        use_temporal_candidates = true;
    }

    int i, j, dir;
    int start_is, end_is, start_js, end_js;
    int start_i, start_j;
    float i_lower_limit = bsz - psz + 1.0f;
    float i_upper_limit = bsz + dis->h - 1.0f;
    float j_lower_limit = bsz - psz + 1.0f;
    float j_upper_limit = bsz + dis->w - 1.0f;
    float dUx, dUy, i_I1, j_I1, w00, w01, w10, w11, dx, dy;

#define INIT_BILINEAR_WEIGHTS(Ux, Uy) \
    i_I1 = min(max(i + Uy + bsz, i_lower_limit), i_upper_limit); \
    j_I1 = min(max(j + Ux + bsz, j_lower_limit), j_upper_limit); \
    { \
        float di = i_I1 - floor(i_I1); \
        float dj = j_I1 - floor(j_I1); \
        w11 = di       * dj; \
        w10 = di       * (1 - dj); \
        w01 = (1 - di) * dj; \
        w00 = (1 - di) * (1 - dj); \
    }

#define COMPUTE_SSD(dst, Ux, Uy)                                                                                       \
    INIT_BILINEAR_WEIGHTS(Ux, Uy);                                                                                     \
    if (dis->use_mean_normalization)                                                                                   \
        dst = computeSSDMeanNorm(I0_ptr + i * dis->w + j, I1_ptr + (int)i_I1 * w_ext + (int)j_I1, dis->w, w_ext, w00,  \
                                 w01, w10, w11, psz);                                                                  \
    else                                                                                                               \
        dst = computeSSD(I0_ptr + i * dis->w + j, I1_ptr + (int)i_I1 * w_ext + (int)j_I1, dis->w, w_ext, w00, w01,     \
                         w10, w11, psz);

    // num_iter似乎是2
    int num_inner_iter = (int)floor(dis->grad_descent_iter / (float)num_iter);
    for (int iter = 0; iter < num_iter; iter++)
    {
        // 分别是正着来一次和反着来一次 不知道为啥要两次？
        if (iter % 2 == 0)
        {
            dir = 1;
            start_is = min(range.start * stripe_sz, hs);
            end_is = min(range.end * stripe_sz, hs);
            start_js = 0;
            end_js = dis->ws;
            start_i = start_is * dis->patch_stride;
            start_j = 0;
        }
        else
        {
            dir = -1;
            start_is = min(range.end * stripe_sz, hs) - 1;
            end_is = min(range.start * stripe_sz, hs) - 1;
            start_js = dis->ws - 1;
            end_js = -1;
            start_i = start_is * dis->patch_stride;
            start_j = (dis->ws - 1) * dis->patch_stride;
        }

        i = start_i;
        for (int is = start_is; dir * is < dir * end_is; is += dir)
        {
            j = start_j;
            for (int js = start_js; dir * js < dir * end_js; js += dir)
            {
                if (iter == 0)
                {
                    //Sx Sy是一块重复利用的区域
                    /* Using result form the previous pyramid level as the very first approximation: */
                    Sx_ptr[is * dis->ws + js] = Ux_ptr[(i + psz2) * dis->w + j + psz2];
                    Sy_ptr[is * dis->ws + js] = Uy_ptr[(i + psz2) * dis->w + j + psz2];
                }

                float min_SSD = INF, cur_SSD;
                if (use_temporal_candidates || dis->use_spatial_propagation)
                {
                    COMPUTE_SSD(min_SSD, Sx_ptr[is * dis->ws + js], Sy_ptr[is * dis->ws + js]);
                }

                if (use_temporal_candidates)
                {
                    /* Try temporal candidates (vectors from the initial flow field that was passed to the function) */
                    COMPUTE_SSD(cur_SSD, initial_Ux_ptr[(i + psz2) * dis->w + j + psz2],
                                initial_Uy_ptr[(i + psz2) * dis->w + j + psz2]);
                    if (cur_SSD < min_SSD)
                    {
                        min_SSD = cur_SSD;
                        Sx_ptr[is * dis->ws + js] = initial_Ux_ptr[(i + psz2) * dis->w + j + psz2];
                        Sy_ptr[is * dis->ws + js] = initial_Uy_ptr[(i + psz2) * dis->w + j + psz2];
                    }
                }

                if (dis->use_spatial_propagation)
                {
                    /* Try spatial candidates: */
                    if (dir * js > dir * start_js)
                    {
                        COMPUTE_SSD(cur_SSD, Sx_ptr[is * dis->ws + js - dir], Sy_ptr[is * dis->ws + js - dir]);
                        if (cur_SSD < min_SSD)
                        {
                            min_SSD = cur_SSD;
                            Sx_ptr[is * dis->ws + js] = Sx_ptr[is * dis->ws + js - dir];
                            Sy_ptr[is * dis->ws + js] = Sy_ptr[is * dis->ws + js - dir];
                        }
                    }
                    /* Flow vectors won't actually propagate across different stripes, which is the reason for keeping
                     * the number of stripes constant. It works well enough in practice and doesn't introduce any
                     * visible seams.
                     */
                    if (dir * is > dir * start_is)
                    {
                        COMPUTE_SSD(cur_SSD, Sx_ptr[(is - dir) * dis->ws + js], Sy_ptr[(is - dir) * dis->ws + js]);
                        if (cur_SSD < min_SSD)
                        {
                            min_SSD = cur_SSD;
                            Sx_ptr[is * dis->ws + js] = Sx_ptr[(is - dir) * dis->ws + js];
                            Sy_ptr[is * dis->ws + js] = Sy_ptr[(is - dir) * dis->ws + js];
                        }
                    }
                }

                /* Use the best candidate as a starting point for the gradient descent: */
                float cur_Ux = Sx_ptr[is * dis->ws + js];
                float cur_Uy = Sy_ptr[is * dis->ws + js];

                /* Computing the inverse of the structure tensor: */
                float detH = xx_ptr[is * dis->ws + js] * yy_ptr[is * dis->ws + js] -
                             xy_ptr[is * dis->ws + js] * xy_ptr[is * dis->ws + js];
                if (abs(detH) < EPS)
                    detH = EPS;
                float invH11 = yy_ptr[is * dis->ws + js] / detH;
                float invH12 = -xy_ptr[is * dis->ws + js] / detH;
                float invH22 = xx_ptr[is * dis->ws + js] / detH;
                float prev_SSD = INF, SSD;
                float x_grad_sum = x_ptr[is * dis->ws + js];
                float y_grad_sum = y_ptr[is * dis->ws + js];

                for (int t = 0; t < num_inner_iter; t++)
                {
                    INIT_BILINEAR_WEIGHTS(cur_Ux, cur_Uy);
                    if (dis->use_mean_normalization)
                        SSD = processPatchMeanNorm(dUx, dUy,
                                I0_ptr  + i * dis->w + j, I1_ptr + (int)i_I1 * w_ext + (int)j_I1,
                                I0x_ptr + i * dis->w + j, I0y_ptr + i * dis->w + j,
                                dis->w, w_ext, w00, w01, w10, w11, psz,
                                x_grad_sum, y_grad_sum);
                    else
                        SSD = processPatch(dUx, dUy,
                                I0_ptr  + i * dis->w + j, I1_ptr + (int)i_I1 * w_ext + (int)j_I1,
                                I0x_ptr + i * dis->w + j, I0y_ptr + i * dis->w + j,
                                dis->w, w_ext, w00, w01, w10, w11, psz);

                    dx = invH11 * dUx + invH12 * dUy;
                    dy = invH12 * dUx + invH22 * dUy;
                    cur_Ux -= dx;
                    cur_Uy -= dy;

                    /* Break when patch distance stops decreasing */
                    if (SSD >= prev_SSD)
                        break;
                    prev_SSD = SSD;
                }

                /* If gradient descent converged to a flow vector that is very far from the initial approximation
                 * (more than patch size) then we don't use it. Noticeably improves the robustness.
                 */
                if (norm(Vec2f(cur_Ux - Sx_ptr[is * dis->ws + js], cur_Uy - Sy_ptr[is * dis->ws + js])) <= psz)
                {
                    Sx_ptr[is * dis->ws + js] = cur_Ux;
                    Sy_ptr[is * dis->ws + js] = cur_Uy;
                }
                j += dir * dis->patch_stride;
            }
            i += dir * dis->patch_stride;
        }
    }
#undef INIT_BILINEAR_WEIGHTS
#undef COMPUTE_SSD
}

```

