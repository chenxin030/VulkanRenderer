// hiz_build.slang — Phase 5: Hi-Z depth pyramid builder for vkrEngine
// Each dispatch builds one mip level: reads from hizIn, writes to hizOut (storage images)
// For mip 0, reads from depthTex; for mip > 0, reads from hizIn (previous level)

[[vk::binding(0, 0)]]
RWTexture2D<float> hizOut; // current mip level (storage write)

[[vk::binding(1, 0)]]
RWTexture2D<float> hizIn; // previous mip level (storage read) — unused for mip 0

[[vk::binding(2, 0)]]
Sampler2D<float> depthTex; // GBuffer depth (only for mip 0)

struct PushConst
{
    uint2 dstSize;
    uint mipLevel; // 0 = sample depthTex; >0 = sample hizIn
    uint pad;
};

[[vk::push_constant]]
PushConst pc;

[numthreads(16, 16, 1)]
[shader("compute")]
void compMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    uint2 dstCoord = dispatchThreadID.xy;
    if (dstCoord.x >= pc.dstSize.x || dstCoord.y >= pc.dstSize.y)
        return;

    uint2 srcBase = dstCoord * 2u;
    float d00, d01, d10, d11;

    if (pc.mipLevel == 0u)
    {
        float2 invSize = 1.0 / float2(pc.dstSize * 2u);
        float2 uv00 = (float2(srcBase) + 0.5) * invSize;
        float2 uv01 = (float2(srcBase.x + 1u, srcBase.y) + 0.5) * invSize;
        float2 uv10 = (float2(srcBase.x, srcBase.y + 1u) + 0.5) * invSize;
        float2 uv11 = (float2(srcBase.x + 1u, srcBase.y + 1u) + 0.5) * invSize;
        d00 = depthTex.SampleLevel(uv00, 0);
        d01 = depthTex.SampleLevel(uv01, 0);
        d10 = depthTex.SampleLevel(uv10, 0);
        d11 = depthTex.SampleLevel(uv11, 0);
    }
    else
    {
        d00 = hizIn[uint2(srcBase.x, srcBase.y)];
        d01 = hizIn[uint2(srcBase.x + 1u, srcBase.y)];
        d10 = hizIn[uint2(srcBase.x, srcBase.y + 1u)];
        d11 = hizIn[uint2(srcBase.x + 1u, srcBase.y + 1u)];
    }

    float minDepth = min(min(d00, d01), min(d10, d11));
    hizOut[dstCoord] = minDepth;
}
