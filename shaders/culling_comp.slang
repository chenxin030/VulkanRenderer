struct SceneUBO
{
    float4x4 projection;
    float4x4 view;
    float3 camPos;
};
[[vk::binding(0, 0)]]
ConstantBuffer<SceneUBO> sceneUbo;

struct InstanceData
{
    float4x4 model;
    float4 color;
};
[[vk::binding(1, 0)]]
StructuredBuffer<InstanceData> instanceBuffer;

struct DrawCommand
{
    uint indexCount;
    uint instanceCount;
    uint firstIndex;
    int vertexOffset;
    uint firstInstance;
};
[[vk::binding(2, 0)]]
RWStructuredBuffer<DrawCommand> drawCommands;

[[vk::binding(3, 0)]]
RWStructuredBuffer<uint> visibleIndices;

struct CullingStats
{
    uint totalCount;
    uint visibleCount;
    float gpuMs;
    float frameMs;
};
[[vk::binding(4, 0)]]
RWStructuredBuffer<CullingStats> stats;

[[vk::binding(5, 0)]]
RWStructuredBuffer<uint> visibleCountBuffer;

struct CullingParams
{
    float4 frustumPlanes[6];
    float4 aabbMin;
    float4 aabbMax;
    float4 hiZInfo; // x: width, y: height, z: mipCount, w: depthBias
    uint totalInstances;
    uint useCulling;
    uint pad2;
    uint pad3;
};
[[vk::binding(6, 0)]]
ConstantBuffer<CullingParams> params;

[[vk::binding(7, 0)]]
Texture2D<float> hiZTexture;
[[vk::binding(8, 0)]]
SamplerState hiZSampler;

bool sphereInsidePlane(float3 center, float radius, float4 plane)
{
    return dot(plane.xyz, center) + plane.w + radius > 0.0;
}

bool sphereInsideFrustum(float3 center, float radius)
{
    [unroll]
    for (int i = 0; i < 6; ++i)
    {
        if (!sphereInsidePlane(center, radius, params.frustumPlanes[i]))
        {
            return false;
        }
    }
    return true;
}

float4 projectToClip(float3 worldPos)
{
    float4 viewPos = mul(sceneUbo.view, float4(worldPos, 1.0));
    return mul(sceneUbo.projection, viewPos);
}

float sampleHiZ(float2 uv, float lod)
{
    return hiZTexture.SampleLevel(hiZSampler, uv, lod);
}

bool occlusionTest(float3 center, float radius)
{
    float4 clip = projectToClip(center);
    if (clip.w <= 0.0)
    {
        return true;
    }

    float3 ndc = clip.xyz / clip.w;
    float2 uv = ndc.xy * 0.5 + 0.5;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0)
    {
        return true;
    }

    float radiusPx = max(1.0, radius * params.hiZInfo.x / max(clip.w, 1e-3));
    float mip = clamp(floor(log2(radiusPx)), 0.0, params.hiZInfo.z - 1.0);

    float maxDepth = sampleHiZ(uv, mip);
    float projectedDepth = ndc.z;
    float radiusProj = radius / max(clip.w, 1e-3);
    float depthTest = projectedDepth - radiusProj;
    return depthTest <= maxDepth + params.hiZInfo.w;
}

[numthreads(64, 1, 1)]
[shader("compute")]
void compMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint index = dispatchThreadID.x;
    if (index >= params.totalInstances)
    {
        return;
    }

    if (index == 0)
    {
        stats[0].totalCount = params.totalInstances;
    }

    if (params.useCulling == 0)
    {
        visibleIndices[index] = index;
        if (index == 0)
        {
            visibleCountBuffer[0] = params.totalInstances;
            drawCommands[0].instanceCount = params.totalInstances;
            stats[0].visibleCount = params.totalInstances;
        }
        return;
    }

    float4x4 model = instanceBuffer[index].model;
    float3 center = mul(model, float4(0.0, 0.0, 0.0, 1.0)).xyz;
    float3 extents = 0.5 * (params.aabbMax.xyz - params.aabbMin.xyz);
    float radius = length(extents);

    bool inside = sphereInsideFrustum(center, radius);
    bool visible = inside && occlusionTest(center, radius);

    if (visible)
    {
        uint dstIndex = 0;
        InterlockedAdd(drawCommands[0].instanceCount, 1, dstIndex);
        visibleIndices[dstIndex] = index;
        InterlockedAdd(visibleCountBuffer[0], 1);
    }
}
