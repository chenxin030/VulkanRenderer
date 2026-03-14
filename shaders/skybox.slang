/* Copyright (c) 2025, Sascha Willems
 *
 * SPDX-License-Identifier: MIT
 *
 */

struct VSOutput
{
    float4 Pos : SV_POSITION;
    float2 Ndc;
};

struct SkyboxUBO
{
    float4x4 invProjection;
    float4x4 invView;
};
[[vk::binding(0, 0)]]
ConstantBuffer<SkyboxUBO> skyboxUbo;

struct VSInput
{
    float3 Pos;
    float3 Normal;
    float2 UV;
};

struct ParamsUBO
{
    float exposure;
    float gamma;
};
[[vk::binding(1, 0)]]
ConstantBuffer<ParamsUBO> paramsUbo;

[[vk::binding(2, 0)]]
SamplerCube samplerEnv;

// From http://filmicworlds.com/blog/filmic-tonemapping-operators/
float3 Uncharted2Tonemap(float3 color)
{
    float A = 0.15;
    float B = 0.50;
    float C = 0.10;
    float D = 0.20;
    float E = 0.02;
    float F = 0.30;
    float W = 11.2;
    return ((color * (A * color + C * B) + D * E) / (color * (A * color + B) + D * F)) - E / F;
}

[shader("vertex")]
VSOutput vertMain(VSInput input)
{
    VSOutput output;
    float2 ndc = input.Pos.xy;
    output.Pos = float4(ndc, 0.0f, 1.0f);
    output.Ndc = ndc;
    return output;
}

[shader("fragment")]
float4 fragMain(VSOutput input) : SV_TARGET
{
    float4 clip = float4(input.Ndc, 1.0f, 1.0f);
    float4 viewPos = mul(skyboxUbo.invProjection, clip);
    viewPos.xyz /= viewPos.w;

    float3 dirView = normalize(viewPos.xyz);
    float3 dirWorld = normalize(mul((float3x3)skyboxUbo.invView, dirView));

    float3 color = samplerEnv.Sample(dirWorld).rgb;

    // Tone mapping
    color = Uncharted2Tonemap(color * paramsUbo.exposure);
    color = color * (1.0f / Uncharted2Tonemap((11.2f).xxx));
    // Gamma correction
    color = pow(color, (1.0f / paramsUbo.gamma).xxx);

    return float4(color, 1.0);
}
