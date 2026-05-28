struct SceneUBO
{
    float4x4 projection;
    float4x4 view;
    float4x4 invProjection;
    float4x4 invView;
    float4 camPos;
};

struct SsaoSettingsUBO
{
    float4x4 invProjection;
    float4x4 invView;
    float4 params0; // x: radius, y: bias, z: intensity, w: enabled
    int4 flags;     // x: debugView
    float4 kernel[16];
};

[[vk::binding(0, 0)]]
ConstantBuffer<SsaoSettingsUBO> settingsUbo;

[[vk::binding(1, 0)]]
Sampler2D<float4> gNormal;

[[vk::binding(2, 0)]]
Sampler2D<float> gDepth;

[[vk::binding(3, 0)]]
Sampler2D<float4> noiseSampler;

struct VSOutput
{
    float4 Pos : SV_POSITION;
    float2 uv;
};

float3 reconstructViewPos(float2 uv, float depth)
{
    float4 clipPos = float4(uv * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
    float4 viewPos = mul(settingsUbo.invProjection, clipPos);
    viewPos /= viewPos.w;
    return viewPos.xyz;
}

float linearizeDepth(float depth, float near, float far)
{
    return near * far / (far + depth * (near - far));
}

[[shader("vertex")]]
VSOutput vertMain(uint vertexID : SV_VertexID)
{
    VSOutput output;

    float2 pos;
    if (vertexID == 0)
    {
        pos = float2(-1.0, -1.0);
    }
    else if (vertexID == 1)
    {
        pos = float2(3.0, -1.0);
    }
    else
    {
        pos = float2(-1.0, 3.0);
    }

    output.Pos = float4(pos, 0.0, 1.0);
    output.uv = pos * 0.5 + 0.5;
    return output;
}

[[shader("fragment")]]
float4 fragMain(VSOutput input) : SV_Target0
{
    if (settingsUbo.flags.x == 5)
    {
        float ao = 0.0f;
        float depth = gDepth.Sample(input.uv);

        if (depth >= 1.0f)
        {
            return float4(1.0, 1.0, 1.0, 1.0);
        }

        float3 viewPos = reconstructViewPos(input.uv, depth);
        float3 normal = gNormal.Sample(input.uv).rgb * 2.0 - 1.0;

        float2 noiseScale = float2(float(viewPos.x) / float(input.uv.x) * 0.25, float(viewPos.y) / float(input.uv.y) * 0.25);
        float3 randomVec = noiseSampler.Sample(noiseScale).xyz * 2.0 - 1.0;

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

            float4 offset = mul(settingsUbo.invProjection, float4(samplePos, 1.0));
            offset.xyz /= offset.w;
            offset.xyz = offset.xyz * 0.5 + 0.5;

            float sampleDepth = gDepth.Sample(offset.xy);

            if (sampleDepth >= 1.0f)
            {
                sampleDepth = 0.0f;
            }

            float sampleViewDepth = reconstructViewPos(offset.xy, sampleDepth).z;

            float rangeCheck = smoothstep(0.0, 1.0, radius / abs(viewPos.z - sampleViewDepth + 0.001));
            occlusion += (sampleViewDepth >= samplePos.z + bias ? 1.0 : 0.0) * rangeCheck;
        }

        ao = 1.0 - (occlusion / 16.0);
        ao = pow(ao, intensity);

        return float4(ao, ao, ao, 1.0);
    }

    return float4(1.0, 1.0, 1.0, 1.0);
}
