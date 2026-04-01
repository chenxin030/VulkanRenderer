// Post-processing: Tone Mapping + Chromatic Aberration + Vignette
// Input: HDR scene texture (RGBA16F)
// Output: LDR to swapchain (8-bit)

[[vk::binding(0, 0)]] Sampler2D<float4> sceneTexture;
[[vk::binding(1, 0)]] ConstantBuffer<PostFXSettingsUBO> settings;

struct VSOutput
{
    float4 Pos : SV_POSITION;
    float2 uv;
};

[[shader("vertex")]]
VSOutput vertMain(uint vertexID : SV_VertexID)
{
    VSOutput output;
    float2 pos;
    if (vertexID == 0) pos = float2(-1.0, -1.0);
    else if (vertexID == 1) pos = float2(3.0, -1.0);
    else pos = float2(-1.0, 3.0);
    output.Pos = float4(pos, 0.0, 1.0);
    output.uv = pos * 0.5 + 0.5;
    return output;
}

float3 ACESFilmic(float3 x)
{
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0f, 1.0f);
}

float3 Reinhard(float3 x)
{
    return x / (x + 1.0f);
}

float3 ApplyToneMapping(float3 color, int mode)
{
    color *= settings.exposure;
    if (mode == 0)
        color = Reinhard(color);
    else
        color = ACESFilmic(color);
    return pow(color, 1.0f / float3(settings.gamma));
}

[[shader("fragment")]]
float4 fragMain(VSOutput input) : SV_Target0
{
    float2 uv = input.uv;

    // Chromatic aberration: offset RGB channels
    float4 color;
    if (settings.chromaticAberrationEnabled > 0 && settings.chromaticAberration > 0.0f)
    {
        float2 center = float2(0.5f, 0.5f);
        float2 dir = uv - center;
        float dist = length(dir);
        float2 offset = dir * settings.chromaticAberration * dist;

        float r = sceneTexture.Sample(float2(uv.x + offset.x, uv.y + offset.y)).r;
        float g = sceneTexture.Sample(uv).g;
        float b = sceneTexture.Sample(float2(uv.x - offset.x, uv.y - offset.y)).b;
        float a = sceneTexture.Sample(uv).a;
        color = float4(r, g, b, a);
    }
    else
    {
        color = sceneTexture.Sample(uv);
    }

    // Tone mapping
    float3 result = ApplyToneMapping(color.rgb, settings.toneMappingMode);

    // Vignette
    if (settings.vignetteEnabled > 0)
    {
        float2 center = float2(0.5f, 0.5f);
        float dist = length(uv - center);
        float vig = 1.0f - smoothstep(0.3f, 0.9f, dist);
        result *= lerp(1.0f - settings.vignetteIntensity, 1.0f, vig);
    }

    return float4(result, color.a);
}

struct PostFXSettingsUBO
{
    float exposure;
    float gamma;
    float bloomThreshold;
    float bloomIntensity;
    float chromaticAberration;
    float vignetteIntensity;
    int toneMappingMode;
    int bloomEnabled;
    int chromaticAberrationEnabled;
    int vignetteEnabled;
    int padding0;
    int padding1;
    int padding2;
};
