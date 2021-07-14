- prepareBuffers 部分
```cpp
void DISOpticalFlowImpl::prepareBuffers(Mat &I0, Mat &I1, Mat &flow, bool use_flow)
{
    /* CV_INSTRUMENT_REGION(); */

    // 一共创建coarsest_scale+1个, idx越小 矩阵越大 相当越处于金字塔底层
    I0s.resize(coarsest_scale + 1); //I0金字塔  
    I1s.resize(coarsest_scale + 1); //I1金字塔
    I1s_ext.resize(coarsest_scale + 1); //I1+border
    I0xs.resize(coarsest_scale + 1); // I0 x梯度
    I0ys.resize(coarsest_scale + 1); // I0 y梯度
    Ux.resize(coarsest_scale + 1);  //每层的光流x
    Uy.resize(coarsest_scale + 1);  //每层的光流y

    Mat flow_uv[2];
    if (use_flow)
    {
        split(flow, flow_uv);
        initial_Ux.resize(coarsest_scale + 1);
        initial_Uy.resize(coarsest_scale + 1);
    }

    int fraction = 1;
    int cur_rows = 0, cur_cols = 0;

    for (int i = 0; i <= coarsest_scale; i++)
    {
        /* Avoid initializing the pyramid levels above the finest scale, as they won't be used anyway */
        // 只生成finest->coarsest_scale层 因为只计算这部分
        if (i == finest_scale)
        {
            cur_rows = I0.rows / fraction;
            cur_cols = I0.cols / fraction;
            // I0 I1都是原图的金字塔缩放
            I0s[i].create(cur_rows, cur_cols);
            resize(I0, I0s[i], I0s[i].size(), 0.0, 0.0, INTER_AREA);   
            I1s[i].create(cur_rows, cur_cols);
            resize(I1, I1s[i], I1s[i].size(), 0.0, 0.0, INTER_AREA);

            /* These buffers are reused in each scale so we initialize them once on the finest scale: */
            Sx.create(cur_rows / patch_stride, cur_cols / patch_stride);
            Sy.create(cur_rows / patch_stride, cur_cols / patch_stride);
            I0xx_buf.create(cur_rows / patch_stride, cur_cols / patch_stride);
            I0yy_buf.create(cur_rows / patch_stride, cur_cols / patch_stride);
            I0xy_buf.create(cur_rows / patch_stride, cur_cols / patch_stride);
            I0x_buf.create(cur_rows / patch_stride, cur_cols / patch_stride);
            I0y_buf.create(cur_rows / patch_stride, cur_cols / patch_stride);

            I0xx_buf_aux.create(cur_rows, cur_cols / patch_stride);
            I0yy_buf_aux.create(cur_rows, cur_cols / patch_stride);
            I0xy_buf_aux.create(cur_rows, cur_cols / patch_stride);
            I0x_buf_aux.create(cur_rows, cur_cols / patch_stride);
            I0y_buf_aux.create(cur_rows, cur_cols / patch_stride);

            U.create(cur_rows, cur_cols);
        }
        else if (i > finest_scale)
        {
            cur_rows = I0s[i - 1].rows / 2;
            cur_cols = I0s[i - 1].cols / 2;
            I0s[i].create(cur_rows, cur_cols);
            resize(I0s[i - 1], I0s[i], I0s[i].size(), 0.0, 0.0, INTER_AREA);
            I1s[i].create(cur_rows, cur_cols);
            resize(I1s[i - 1], I1s[i], I1s[i].size(), 0.0, 0.0, INTER_AREA);
        }

        if (i >= finest_scale)
        {
            I1s_ext[i].create(cur_rows + 2 * border_size, cur_cols + 2 * border_size);
            copyMakeBorder(I1s[i], I1s_ext[i], border_size, border_size, border_size, border_size, BORDER_REPLICATE);
            I0xs[i].create(cur_rows, cur_cols);
            I0ys[i].create(cur_rows, cur_cols);
            spatialGradient(I0s[i], I0xs[i], I0ys[i]);  // 用sobel算空间梯度
            Ux[i].create(cur_rows, cur_cols);
            Uy[i].create(cur_rows, cur_cols);
            variational_refinement_processors[i]->setAlpha(variational_refinement_alpha);
            variational_refinement_processors[i]->setDelta(variational_refinement_delta);
            variational_refinement_processors[i]->setGamma(variational_refinement_gamma);
            variational_refinement_processors[i]->setSorIterations(5);
            variational_refinement_processors[i]->setFixedPointIterations(variational_refinement_iter);

            if (use_flow)
            {
                resize(flow_uv[0], initial_Ux[i], Size(cur_cols, cur_rows));
                initial_Ux[i] /= fraction;
                resize(flow_uv[1], initial_Uy[i], Size(cur_cols, cur_rows));
                initial_Uy[i] /= fraction;
            }
        }

        fraction *= 2;
    }
}

```
