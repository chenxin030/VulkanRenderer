/* Copyright (c) 2025, Sascha Willems
 *
 * SPDX-License-Identifier: MIT
 *
 */

struct VSInput
{
    float3 Pos;
};

struct VSOutput
{
    float4 Pos : SV_POSITION;
    float3 Dir;
};

struct PushConsts
{
    float4x4 mvp;
};
[[vk::push_constant]]
PushConsts pc;

[[vk::binding(0, 0)]]
Sampler2D samplerEquirect;

static const float PI = 3.1415926535897932384626433832795;

float2 sampleSphericalMap(float3 v)
{
    float2 uv;
    uv.x = atan2(v.z, v.x) / (2.0 * PI) + 0.5;
    uv.y = asin(clamp(v.y, -1.0, 1.0)) / PI + 0.5;
    return uv;
}

[shader("vertex")]
VSOutput vertMain(VSInput input)
{
    VSOutput output;
    output.Dir = input.Pos;
    output.Pos = mul(pc.mvp, float4(input.Pos.xyz, 1.0));
    return output;
}

[shader("fragment")]
float4 fragMain(VSOutput input) : SV_TARGET
{
    float3 dir = normalize(input.Dir);
    float2 uv = sampleSphericalMap(dir);
    float3 color = samplerEquirect.Sample(uv).rgb;
    return float4(color, 1.0);
}
