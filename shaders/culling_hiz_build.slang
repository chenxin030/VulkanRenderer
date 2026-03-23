[[vk::binding(0, 0)]]
RWTexture2D<float> outHiZ;

[[vk::binding(1, 0)]]
RWTexture2D<float> unusedOut;

[[vk::binding(2, 0)]]
Texture2D<float> srcDepth;
[[vk::binding(3, 0)]]
SamplerState srcSampler;

[numthreads(8, 8, 1)]
[shader("compute")]
void compMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 dstCoord = dispatchThreadID.xy;
    uint2 dstSize;
    outHiZ.GetDimensions(dstSize.x, dstSize.y);

    if (dstCoord.x >= dstSize.x || dstCoord.y >= dstSize.y)
    {
        return;
    }

    float2 texelSize = 1.0 / float2(dstSize);
    float2 baseUV = (float2(dstCoord) + 0.5) * texelSize;

    // 保守遮挡：取 2x2 区域最大深度构建 max-depth Hi-Z。
    float d0 = srcDepth.SampleLevel(srcSampler, baseUV, 0.0);
    float d1 = srcDepth.SampleLevel(srcSampler, baseUV + float2(texelSize.x, 0.0), 0.0);
    float d2 = srcDepth.SampleLevel(srcSampler, baseUV + float2(0.0, texelSize.y), 0.0);
    float d3 = srcDepth.SampleLevel(srcSampler, baseUV + texelSize, 0.0);
    float maxDepth = max(max(d0, d1), max(d2, d3));

    outHiZ[dstCoord] = maxDepth;
}