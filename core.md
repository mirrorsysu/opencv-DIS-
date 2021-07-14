# DIS光流 - opencv算法

- calc部分
```cpp
void DISOpticalFlowImpl::calc(InputArray I0, InputArray I1, InputOutputArray flow)
{
    Mat I0Mat = I0.getMat();
    Mat I1Mat = I1.getMat();
    bool use_input_flow = false;
    if (flow.sameSize(I0) && flow.depth() == CV_32F && flow.channels() == 2)
        use_input_flow = true;
    else
        flow.create(I1Mat.size(), CV_32FC2);
    Mat flowMat = flow.getMat();

    // 金字塔的个数
    coarsest_scale = min((int)(log(max(I0Mat.cols, I0Mat.rows) / (4.0 * patch_size)) / log(2.0) + 0.5), /* Original code search for maximal movement of width/4 */
                         (int)(log(min(I0Mat.cols, I0Mat.rows) / patch_size) / log(2.0)));              /* Deepest pyramid level greater or equal than patch*/

    if (coarsest_scale<0)
        CV_Error(cv::Error::StsBadSize, "The input image must have either width or height >= 12");

    if (coarsest_scale<finest_scale)
    {
        // choose the finest level based on coarsest level.
        // Refs: https://github.com/tikroeger/OF_DIS/blob/2c9f2a674f3128d3a41c10e41cc9f3a35bb1b523/run_dense.cpp#L239
        int original_img_width = I0.size().width;
        autoSelectPatchSizeAndScales(original_img_width);
    }

    int num_stripes = getNumThreads();

    // 初始化矩阵
    prepareBuffers(I0Mat, I1Mat, flowMat, use_input_flow);
    Ux[coarsest_scale].setTo(0.0f);
    Uy[coarsest_scale].setTo(0.0f);

    for (int i = coarsest_scale; i >= finest_scale; i--) // 从顶层往下
    {
        CV_TRACE_REGION("coarsest_scale_iteration");
        w = I0s[i].cols;
        h = I0s[i].rows;
        ws = 1 + (w - patch_size) / patch_stride;
        hs = 1 + (h - patch_size) / patch_stride;

        precomputeStructureTensor(I0xx_buf, I0yy_buf, I0xy_buf, I0x_buf, I0y_buf, I0xs[i], I0ys[i]);
        if (use_spatial_propagation)
        {
            /* Use a fixed number of stripes regardless the number of threads to make inverse search
             * with spatial propagation reproducible
             */
            parallel_for_(Range(0, 8), PatchInverseSearch_ParBody(*this, 8, hs, Sx, Sy, Ux[i], Uy[i], I0s[i],
                                                                  I1s_ext[i], I0xs[i], I0ys[i], 2, i));
        }
        else
        {
            parallel_for_(Range(0, num_stripes),
                          PatchInverseSearch_ParBody(*this, num_stripes, hs, Sx, Sy, Ux[i], Uy[i], I0s[i], I1s_ext[i],
                                                     I0xs[i], I0ys[i], 1, i));
        }

        parallel_for_(Range(0, num_stripes),
                      Densification_ParBody(*this, num_stripes, I0s[i].rows, Ux[i], Uy[i], Sx, Sy, I0s[i], I1s[i]));
        if (variational_refinement_iter > 0)
            variational_refinement_processors[i]->calcUV(I0s[i], I1s[i], Ux[i], Uy[i]);

        if (i > finest_scale)
        {
            resize(Ux[i], Ux[i - 1], Ux[i - 1].size());
            resize(Uy[i], Uy[i - 1], Uy[i - 1].size());
            Ux[i - 1] *= 2;
            Uy[i - 1] *= 2;
        }
    }
    Mat uxy[] = {Ux[finest_scale], Uy[finest_scale]};
    merge(uxy, 2, U);
    resize(U, flowMat, flowMat.size());
    flowMat *= 1 << finest_scale;
}
```

- 