// ssao.slang — Phase 4: SSAO compute pass for vkrEngine
// Full-screen triangle, reads GBuffer normal+depth, outputs ambient occlusion
//
// Debug views (set via flags.x):
//   0 = normal SSAO
//   1 = linearized view-space depth (heatmap)
//   2 = world-space normal
//   3 = noise pattern
//   4 = occlusion hit count (green=many occluders → dark AO)
//   5 = depth discontinuity (range check)
//   6 = reconstructed viewPos.z

struct SSAOSettingsUBO
{
    float4x4 invProjection; // for NDC→view reconstruction
    float4x4 invView;       // for world→view normal transform
    float4x4 projection;    // for view→NDC forward projection of samples
    float4 params0;         // x: radius, y: bias, z: intensity, w: enabled
    int4 flags;             // x: debugView (0-6)
    int pad1, pad2, pad3;
    float4 kernel[16];
};
[[vk::binding(0, 0)]]
ConstantBuffer<SSAOSettingsUBO> settingsUbo;

[[vk::binding(1, 0)]]
Sampler2D<float4> gNormal;

[[vk::binding(2, 0)]]
Sampler2D<float> gDepth;

[[vk::binding(3, 0)]]
Sampler2D<float4> noiseSampler;

struct VSOutput
{
    float4 Pos : SV_POSITION;
    float2 uv : TEXCOORD0;
};

float3 reconstructViewPos(float2 uv, float depth)
{
    // Vulkan NDC Z is [0,1], NOT [-1,1]. Use depth directly, no *2-1 mapping.
    float4 clipPos = float4(uv * 2.0 - 1.0, depth, 1.0);
    float4 viewPos = mul(settingsUbo.invProjection, clipPos);
    viewPos /= viewPos.w;
    return viewPos.xyz;
}

[shader("vertex")]
VSOutput vertMain(uint vertexID: SV_VertexID)
{
    VSOutput output;

    float2 pos;
    if (vertexID == 0u)
        pos = float2(-1.0, -1.0);
    else if (vertexID == 1u)
        pos = float2(3.0, -1.0);
    else
        pos = float2(-1.0, 3.0);

    output.Pos = float4(pos, 0.0, 1.0);
    output.uv = pos * 0.5 + 0.5;
    return output;
}

[shader("fragment")]
float4 fragMain(VSOutput input) : SV_Target0
{
    float depth = gDepth.Sample(input.uv);
    int debugView = settingsUbo.flags.x;

    // Skip sky / far plane
    if (depth >= 1.0)
    {
        if (debugView == 0)
            return float4(1.0, 1.0, 1.0, 1.0);
        if (debugView == 1)
            return float4(0.0, 0.0, 1.0, 1.0); // blue = sky
        if (debugView == 2)
            return float4(0.5, 0.5, 0.5, 1.0);
        if (debugView == 3)
            return float4(0.0, 0.0, 0.0, 1.0);
        if (debugView == 4)
            return float4(0.0, 0.0, 0.0, 1.0);
        if (debugView == 5)
            return float4(0.0, 0.0, 0.0, 1.0);
        if (debugView == 6)
            return float4(1.0, 0.0, 0.0, 1.0); // red = invalid
    }

    float3 viewPos = reconstructViewPos(input.uv, depth);
    // normal from GBuffer is in WORLD space; transform to VIEW space for consistency
    float3 normalWS = gNormal.Sample(input.uv).rgb * 2.0 - 1.0;
    float3 normal = normalize(mul(normalWS, (float3x3)settingsUbo.invView));

    // ---- Debug view 1: linearized view-space depth ----
    if (debugView == 1)
    {
        float linearZ = saturate(-viewPos.z / 200.0);
        return float4(linearZ, 0.0, 1.0 - linearZ, 1.0);
    }

    // ---- Debug view 2: view-space normal ----
    if (debugView == 2)
    {
        return float4(normal * 0.5 + 0.5, 1.0);
    }

    // ---- Debug view 6: reconstructed viewPos.z raw ----
    if (debugView == 6)
    {
        float z = -viewPos.z;
        // Red if z < 0 (behind camera), green/yellow gradient otherwise
        if (z < 0.0)
            return float4(1.0, 0.0, 0.0, 1.0);
        float t = saturate(z / 500.0); // 0=very close, 1=500+ units
        return float4(t, 1.0 - t, 0.0, 1.0);
    }

    // Tile noise for rotation — use 100× tiling for good distribution
    float3 randomVec = noiseSampler.SampleLevel(input.uv * 100.0, 0.0).xyz * 2.0 - 1.0;

    // ---- Debug view 3: noise pattern ----
    if (debugView == 3)
    {
        return float4(randomVec * 0.5 + 0.5, 1.0);
    }

    float3 tangent = normalize(randomVec - normal * dot(randomVec, normal));
    float3 bitangent = cross(normal, tangent);
    float3x3 TBN = float3x3(tangent, bitangent, normal);

    float occlusion = 0.0;
    float radius = settingsUbo.params0.x;
    float bias = settingsUbo.params0.y;
    float intensity = settingsUbo.params0.z;

    for (int i = 0; i < 16; ++i)
    {
        float3 sampleDir = mul(settingsUbo.kernel[i].xyz, TBN);
        float3 samplePos = viewPos + sampleDir * radius;

        // Project samplePos (view space → clip space → NDC → UV)
        float4 clipPos = mul(settingsUbo.projection, float4(samplePos, 1.0));
        float3 ndc = clipPos.xyz / clipPos.w;
        float2 sampleUV = ndc.xy * 0.5 + 0.5;

        float sampleDepth = gDepth.SampleLevel(sampleUV, 0.0);

        float sampleViewDepth;
        if (sampleDepth < 1.0)
            sampleViewDepth = reconstructViewPos(sampleUV, sampleDepth).z;
        else
            sampleViewDepth = -1e6;

        float rangeCheck = smoothstep(0.0, 1.0, radius / abs(viewPos.z - sampleViewDepth + 0.001));
        occlusion += (sampleViewDepth >= samplePos.z + bias ? 1.0 : 0.0) * rangeCheck;
    }

    // ---- Debug view 4: occlusion hit count ----
    if (debugView == 4)
    {
        // occlusion ranges 0~16. Green = many occluders (high occlusion → dark AO)
        float t = occlusion / 16.0;
        return float4(1.0 - t, t, 0.0, 1.0);
    }

    // ---- Debug view 5: depth discontinuity (range check sum) ----
    if (debugView == 5)
    {
        // Recompute just the range check sum
        float rangeSum = 0.0;
        for (int i2 = 0; i2 < 16; ++i2)
        {
            float3 sd = mul(settingsUbo.kernel[i2].xyz, TBN);
            float3 sp = viewPos + sd * radius;
            float4 c = mul(settingsUbo.projection, float4(sp, 1.0));
            float3 nd = c.xyz / c.w;
            float2 suv = nd.xy * 0.5 + 0.5;
            float sdepth = gDepth.SampleLevel(suv, 0.0);
            float svd = (sdepth < 1.0) ? reconstructViewPos(suv, sdepth).z : -1e6;
            rangeSum += smoothstep(0.0, 1.0, radius / abs(viewPos.z - svd + 0.001));
        }
        float t2 = rangeSum / 16.0;
        return float4(t2, 0.0, 1.0 - t2, 1.0); // red=large discontinuities
    }

    // ---- Normal SSAO ----
    float ao = 1.0 - (occlusion / 16.0);
    ao = pow(ao, intensity);

    return float4(ao, ao, ao, 1.0);
}
