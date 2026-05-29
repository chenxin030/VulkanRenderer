// Volumetric Lighting - Fragment Shader
// Phase 1: Basic Ray Marching with Beer-Lambert + Henyey-Greenstein

[[vk::binding(0, 0)]]
cbuffer VolumetricParams : register(b0)
{
    float density;
    float scattering;
    float absorption;
    float anisotropicG;
    float stepSize;
    float maxDistance;
    float temporalFactor;
    float shadowStrength;
    float intensity;
    float nearZ;
    float farZ;
    float2 renderScale;
    float2 jitter;
    float historyValid;
};

[[vk::binding(1, 0)]]
Sampler2D<float4> historyTexture;

[[vk::binding(2, 0)]]
Sampler2D<float4> depthTexture;

[[vk::binding(3, 0)]]
Sampler2D<float4> velocityTexture;

static const float3 lightDir = normalize(float3(0.5, -0.8, 0.3));
static const float3 lightColor = float3(1.0, 0.95, 0.85); // Warm sunlight
static const float PI = 3.14159265359;
static const int MAX_STEPS = 64;

float henyeyGreenstein(float cosTheta, float g)
{
    float g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(max(0.0001, 1.0 + g2 - 2.0 * g * cosTheta), 1.5));
}

float beerLambert(float distance, float absorption)
{
    return exp(-absorption * distance);
}

float blueNoise(float2 uv)
{
    return fract(sin(dot(uv, float2(12.9898, 78.233))) * 43758.5453);
}

[shader("fragment")]
float4 fragMain(
    [[vk::location(0)]] float2 uv: TEXCOORD0,
    float4 position: SV_POSITION) : SV_Target
{
    float3 viewPos = depthTexture.Sample(uv).rgb;

    if (viewPos.z >= -nearZ * 0.1)
        return float4(0.0, 0.0, 0.0, 0.0);

    float3 rayDir = normalize(viewPos);
    float3 rayOrigin = float3(0.0);
    float surfaceDist = length(viewPos);
    float dither = blueNoise(uv + jitter) * stepSize;
    float startDist = nearZ + dither;
    float endDist = min(surfaceDist, maxDistance);

    float3 accumulatedLight = float3(0.0);
    float transmittance = 1.0;
    float cosTheta = dot(rayDir, -lightDir);
    float phase = henyeyGreenstein(cosTheta, anisotropicG);

    for (int i = 0; i < MAX_STEPS; i++)
    {
        float currentDist = startDist + float(i) * stepSize;
        if (currentDist > endDist)
            break;
        if (transmittance < 0.01)
            break;

        float3 samplePos = rayOrigin + rayDir * currentDist;
        float extinction = density * absorption;
        float sampleTransmittance = beerLambert(stepSize, extinction);
        float3 inScattering = phase * scattering * lightColor * intensity * transmittance;
        accumulatedLight += inScattering * stepSize;
        transmittance *= sampleTransmittance;
    }

    float2 velocity = velocityTexture.Sample(uv).rg;
    float2 historyUV = uv - velocity;
    float4 history = historyTexture.Sample(historyUV);
    float3 result = accumulatedLight;

    if (historyValid > 0.5)
    {
        float3 historyMin = history.rgb * 0.8;
        float3 historyMax = history.rgb * 1.2;
        float3 clampedHistory = clamp(history.rgb, historyMin, historyMax);
        result = lerp(accumulatedLight, clampedHistory, temporalFactor);
    }

    return float4(result, transmittance > 0.01 ? 1.0 : 0.0);
}
