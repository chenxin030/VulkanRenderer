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
    float4 position; // w is intensity or unused
    float4 color;    // w is intensity
};

struct LightUBO
{
    PointLight lights[4];
};
[[vk::binding(2, 0)]]
ConstantBuffer<LightUBO> lightUbo;

struct VSInput
{
    float3 Pos;
    float3 Normal;
    float2 texCoord; // unused
};

struct VSOutput
{
    float4 Pos : SV_POSITION;
    float3 WorldPos;
    float3 Normal;
    float2 texCoord; // unused
    uint InstanceIndex : SV_InstanceID;
};

static const float PI = 3.14159265359;

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
float3 F_Schlick(float cosTheta, float3 color, float metallic)
{
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), color, metallic);
    float3 F = F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
    return F;
}

// Specular BRDF composition --------------------------------------------
float3 BRDF(float3 L, float3 V, float3 N, float3 color, float metallic, float roughness, float3 lightColor)
{
    // Precalculate vectors and dot products
    float3 H = normalize(V + L);
    float dotNV = clamp(dot(N, V), 0.0, 1.0);
    float dotNL = clamp(dot(N, L), 0.0, 1.0);
    float dotLH = clamp(dot(L, H), 0.0, 1.0);
    float dotNH = clamp(dot(N, H), 0.0, 1.0);

    float3 result = float3(0.0, 0.0, 0.0);

    if (dotNL > 0.0)
    {
        float rroughness = max(0.05, roughness);
        // D = Normal distribution (Distribution of the microfacets)
        float D = D_GGX(dotNH, roughness);
        // G = Geometric shadowing term (Microfacets shadowing)
        float G = G_SchlicksmithGGX(dotNL, dotNV, rroughness);
        // F = Fresnel factor (Reflectance depending on angle of incidence)
        float3 F = F_Schlick(dotNV, color, metallic);

        float3 spec = D * F * G / (4.0 * dotNL * dotNV + 0.001);

        float3 kD = (float3(1.0, 1.0, 1.0) - F) * (1.0 - metallic);
        float3 diffuse = kD * color / PI;

        result = (diffuse + spec) * dotNL * lightColor;
    }

    return result;
}

[shader("vertex")]
VSOutput vertMain(VSInput input, uint instanceID: SV_InstanceID)
{
    VSOutput output;
    PBRInstanceData data = instanceData[instanceID];

    float4 worldPos = mul(data.model, float4(input.Pos, 1.0));
    output.WorldPos = worldPos.xyz;
    output.Normal = normalize(mul((float3x3)data.model, input.Normal));
    output.Pos = mul(sceneUbo.projection, mul(sceneUbo.view, worldPos));
    output.InstanceIndex = instanceID;
    output.texCoord = input.texCoord;
    return output;
}

[shader("fragment")]
float4 fragMain(VSOutput input)
{
    PBRInstanceData data = instanceData[input.InstanceIndex];

    float3 N = normalize(input.Normal);
    float3 V = normalize(sceneUbo.camPos - input.WorldPos);

    float3 Lo = float3(0.0, 0.0, 0.0);
    for (int i = 0; i < 4; i++)
    {
        float3 L = normalize(lightUbo.lights[i].position.xyz - input.WorldPos);

        //float dist = length(lightUbo.lights[i].position.xyz - input.WorldPos);
        //float attenuation = 1.0 / (dist * dist + 0.1);
        //float3 lightIntensity = lightUbo.lights[i].color.xyz * attenuation;

        Lo += BRDF(L, V, N, data.color, data.metallic, data.roughness, lightUbo.lights[i].color.xyz);
    };

    // Low ambient light for high contrast
    float3 color = data.color * 0.02;
    color += Lo;

    // HDR Tonemapping (Reinhard)
    color = color / (color + float3(1.0, 1.0, 1.0));
    // Gamma correct
    color = pow(color, float3(0.4545, 0.4545, 0.4545));

    return float4(color, 1.0);
}
