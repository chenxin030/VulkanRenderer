// Bloom extraction shader
// Samples HDR scene texture and extracts bright pixels for bloom
// Downsampled to quarter resolution externally (1/4 width, 1/4 height)

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

float luminance(float3 c)
{
    return dot(c, float3(0.2126f, 0.7152f, 0.0722f));
}

[[shader("fragment")]]
float4 fragMain(VSOutput input) : SV_Target0
{
    float4 color = sceneTexture.Sample(input.uv);

    if (settings.bloomEnabled == 0)
    {
        return float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    // Apply exposure before threshold
    float3 exposed = color.rgb * settings.exposure;

    // Extract bright pixels above threshold
    float lum = luminance(exposed);
    float weight = max(lum - settings.bloomThreshold, 0.0f);

    // Soft knee - smooth transition around threshold
    float knee = settings.bloomThreshold * 0.5f;
    if (lum > settings.bloomThreshold - knee)
    {
        float x = lum - (settings.bloomThreshold - knee);
        float t = saturate(x / (2.0f * knee + 1e-5f));
        weight = lerp(0.0f, weight, t * t);
    }

    // Scale by bloom intensity
    float3 bloomColor = color.rgb * (weight / max(lum, 1e-5f)) * settings.bloomIntensity;

    return float4(bloomColor, 1.0f);
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
