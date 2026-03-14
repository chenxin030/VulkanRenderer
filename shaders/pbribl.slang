/* Copyright (c) 2025, Sascha Willems
 *
 * SPDX-License-Identifier: MIT
 *
 */

struct SceneUBO
{
    float4x4 projection;
    float4x4 view;
    float3 camPos;
};
[[vk::binding(0, 0)]]
ConstantBuffer<SceneUBO> sceneUbo;

struct PBRInstanceData
{
    float4x4 model;
    float metallic;
    float roughness;
    float3 color;
};
[[vk::binding(1, 0)]]
StructuredBuffer<PBRInstanceData> instanceData;

struct PointLight
{
    float4 position;
    float4 color;
};

struct LightUBO
{
    PointLight lights[4];
};
[[vk::binding(2, 0)]]
ConstantBuffer<LightUBO> lightUbo;

[[vk::binding(3, 0)]]
SamplerCube samplerIrradiance;
[[vk::binding(4, 0)]]
SamplerCube prefilteredMapSampler;
[[vk::binding(5, 0)]]
Sampler2D samplerBRDFLUT;

struct ParamsUBO
{
    float exposure;
    float gamma;
};
[[vk::binding(6, 0)]]
ConstantBuffer<ParamsUBO> paramsUbo;

struct VSInput
{
    float3 Pos;
    float3 Normal;
    float2 UV;
};

struct VSOutput
{
    float4 Pos : SV_POSITION;
    float3 WorldPos;
    float3 Normal;
    uint InstanceIndex : SV_InstanceID;
};

static const float PI = 3.1415926535897932384626433832795;

// From http://filmicgames.com/archives/75
float3 Uncharted2Tonemap(float3 x)
{
    float A = 0.15;
    float B = 0.50;
    float C = 0.10;
    float D = 0.20;
    float E = 0.02;
    float F = 0.30;
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

// Normal Distribution function --------------------------------------
float D_GGX(float dotNH, float roughness)
{
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    float denom = dotNH * dotNH * (alpha2 - 1.0) + 1.0;
    return (alpha2) / (PI * denom * denom);
}

// Geometric Shadowing function --------------------------------------
float G_SchlicksmithGGX(float dotNL, float dotNV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;
    float GL = dotNL / (dotNL * (1.0 - k) + k);
    float GV = dotNV / (dotNV * (1.0 - k) + k);
    return GL * GV;
}

// Fresnel function ----------------------------------------------------
float3 F_Schlick(float cosTheta, float3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}
float3 F_SchlickR(float cosTheta, float3 F0, float roughness)
{
    return F0 + (max((1.0 - roughness).xxx, F0) - F0) * pow(1.0 - cosTheta, 5.0);
}

float3 prefilteredReflection(float3 R, float roughness)
{
    const float MAX_REFLECTION_LOD = 9.0; // todo: param/const
    float lod = roughness * MAX_REFLECTION_LOD;
    float lodf = floor(lod);
    float lodc = ceil(lod);
    float3 a = prefilteredMapSampler.SampleLevel(R, lodf).rgb;
    float3 b = prefilteredMapSampler.SampleLevel(R, lodc).rgb;
    return lerp(a, b, lod - lodf);
}

float3 directLighting(float3 L, float3 V, float3 N, float3 albedo, float metallic, float roughness, float3 radiance)
{
    float3 H = normalize(V + L);
    float dotNL = clamp(dot(N, L), 0.0, 1.0);
    float dotNV = clamp(dot(N, V), 0.0, 1.0);
    float dotNH = clamp(dot(N, H), 0.0, 1.0);
    float dotVH = clamp(dot(V, H), 0.0, 1.0);

    float3 F0 = lerp((0.04).xxx, albedo, metallic);
    float3 F = F_Schlick(dotVH, F0);
    float D = D_GGX(dotNH, roughness);
    float G = G_SchlicksmithGGX(dotNL, dotNV, roughness);
    float3 spec = (D * G * F) / (4.0 * dotNL * dotNV + 0.001);

    float3 kS = F;
    float3 kD = (1.0 - kS) * (1.0 - metallic);
    float3 diffuse = kD * albedo / PI;

    return (diffuse + spec) * radiance * dotNL;
}

[shader("vertex")]
VSOutput vertMain(VSInput input, uint instanceID: SV_InstanceID)
{
    VSOutput output;
    PBRInstanceData data = instanceData[instanceID];

    float4 worldPos = mul(data.model, float4(input.Pos.xyz, 1.0));
    output.WorldPos = worldPos.xyz;
    output.Normal = normalize(mul((float3x3)data.model, input.Normal));
    output.Pos = mul(sceneUbo.projection, mul(sceneUbo.view, worldPos));
    output.InstanceIndex = instanceID;
    return output;
}

[shader("fragment")]
float4 fragMain(VSOutput input) : SV_TARGET
{
    PBRInstanceData data = instanceData[input.InstanceIndex];

    float3 N = normalize(input.Normal);
    float3 V = normalize(sceneUbo.camPos - input.WorldPos);
    float3 R = reflect(-V, N);

    float metallic = data.metallic;
    float roughness = max(data.roughness, 0.04);
    float3 albedo = data.color;

    float3 F0 = lerp((0.04).xxx, albedo, metallic);

    float3 Lo = float3(0.0, 0.0, 0.0);
    for (int i = 0; i < 4; i++)
    {
        float3 lightPos = lightUbo.lights[i].position.xyz;
        float3 Lvec = lightPos - input.WorldPos;
        float3 L = normalize(Lvec);
        float dist = length(Lvec);
        float attenuation = 1.0 / max(dist * dist, 0.01);
        float3 radiance = lightUbo.lights[i].color.xyz * lightUbo.lights[i].color.w * attenuation;
        Lo += directLighting(L, V, N, albedo, metallic, roughness, radiance);
    }

    float2 brdf = samplerBRDFLUT.Sample(float2(max(dot(N, V), 0.0), roughness)).rg;
    float3 reflection = prefilteredReflection(R, roughness).rgb;
    float3 irradiance = samplerIrradiance.Sample(N).rgb;

    // Diffuse based on irradiance
    float3 diffuse = irradiance * albedo;

    float3 F = F_SchlickR(max(dot(N, V), 0.0), F0, roughness);

    // Specular reflectance
    float3 specular = reflection * (F * brdf.x + brdf.y);

    // Ambient part
    float3 kD = (1.0 - F);
    kD *= 1.0 - metallic;
    float3 ambient = (kD * diffuse + specular);

    float3 color = ambient + Lo;

    // Tone mapping
    color = Uncharted2Tonemap(color * paramsUbo.exposure);
    color = color * (1.0f / Uncharted2Tonemap((11.2f).xxx));
    // Gamma correction
    color = pow(color, (1.0f / paramsUbo.gamma).xxx);

    return float4(color, 1.0);
}
