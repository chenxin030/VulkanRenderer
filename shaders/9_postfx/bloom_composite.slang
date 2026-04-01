// Bloom composite shader
// Adds bloom to HDR scene, then applies tone mapping + vignette + chromatic aberration
// Final output: LDR to swapchain

[[vk::binding(0, 0)]] Sampler2D<float4> sceneTexture;
[[vk::binding(1, 0)]] Sampler2D<float4> bloomTexture;
[[vk::binding(2, 0)]] ConstantBuffer<PostFXSettingsUBO> settings;

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

    // Sample scene
    float4 scene = sceneTexture.Sample(uv);

    // Sample bloom with chromatic aberration
    float4 bloom;
    if (settings.chromaticAberrationEnabled > 0 && settings.chromaticAberration > 0.0f)
    {
        float2 center = float2(0.5f, 0.5f);
        float2 dir = uv - center;
        float dist = length(dir);
        float2 offset = dir * settings.chromaticAberration * dist;

        float r = bloomTexture.Sample(float2(uv.x + offset.x, uv.y + offset.y)).r;
        float g = bloomTexture.Sample(uv).g;
        float b = bloomTexture.Sample(float2(uv.x - offset.x, uv.y - offset.y)).b;
        bloom = float4(r, g, b, 1.0f);
    }
    else
    {
        bloom = bloomTexture.Sample(uv);
    }

    // Composite bloom onto scene
    float3 color;
    if (settings.bloomEnabled > 0)
    {
        color = scene.rgb + bloom.rgb * settings.bloomIntensity;
    }
    else
    {
        color = scene.rgb;
    }

    // Tone mapping
    color = ApplyToneMapping(color, settings.toneMappingMode);

    // Vignette
    if (settings.vignetteEnabled > 0)
    {
        float2 center = float2(0.5f, 0.5f);
        float dist = length(uv - center);
        float vig = 1.0f - smoothstep(0.3f, 0.9f, dist);
        color *= lerp(1.0f - settings.vignetteIntensity, 1.0f, vig);
    }

    return float4(color, scene.a);
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
