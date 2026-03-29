struct SceneUBO
{
    float4x4 projection;
    float4x4 view;
    float4x4 invProjection;
    float4x4 invView;
    float4 camPos;
};

struct PointLight
{
    float4 position;
    float4 color;
};

struct LightUBO
{
    PointLight lights[4];
};

struct DeferredSettingsUBO
{
    float4 params0; // x ambient, y exposure, z gamma, w lightScale
    int4 debug;     // x debugView
};

[[vk::binding(0, 0)]]
Sampler2D<float4> gAlbedo;
[[vk::binding(1, 0)]]
Sampler2D<float4> gNormal;
[[vk::binding(2, 0)]]
Sampler2D<float4> gMaterial;
[[vk::binding(3, 0)]]
Sampler2D<float> gDepth;
[[vk::binding(4, 0)]]
ConstantBuffer<SceneUBO> sceneUbo;
[[vk::binding(5, 0)]]
ConstantBuffer<LightUBO> lightUbo;
[[vk::binding(6, 0)]]
ConstantBuffer<DeferredSettingsUBO> settingsUbo;

struct VSOutput
{
    float4 Pos : SV_POSITION;
    float2 uv;
};

static const float PI = 3.14159265359;

float D_GGX(float dotNH, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float denom = dotNH * dotNH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom + 1e-6);
}

float G_SchlickSmith(float dotNL, float dotNV, float roughness)
{
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    float gL = dotNL / (dotNL * (1.0 - k) + k + 1e-6);
    float gV = dotNV / (dotNV * (1.0 - k) + k + 1e-6);
    return gL * gV;
}

float3 F_Schlick(float cosTheta, float3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

float3 reconstructWorldPos(float2 uv, float depth)
{
    float4 clip = float4(uv * 2.0 - 1.0, depth, 1.0);
    float4 viewPos = mul(sceneUbo.invProjection, clip);
    viewPos /= max(viewPos.w, 1e-6);
    float4 worldPos = mul(sceneUbo.invView, viewPos);
    return worldPos.xyz;
}

float linearizeDepth(float2 uv, float depth)
{
    // Reconstruct world position and use camera distance for robust debug visualization.
    // This avoids sign/convention issues in view-space Z across APIs/projection setups.
    float3 worldPos = reconstructWorldPos(uv, depth);
    float viewDepth = distance(sceneUbo.camPos.xyz, worldPos);

    // Debug remap range: scene spheres are roughly within 0~30 units from camera.
    const float debugDepthRange = 30.0;
    float linear01 = saturate(viewDepth / debugDepthRange);
    return linear01;
}

float3 tonemapReinhard(float3 color)
{
    return color / (color + 1.0);
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
    float2 uv = input.uv;

    float4 albedoData = gAlbedo.Sample(uv);
    float4 normalData = gNormal.Sample(uv);
    float4 materialData = gMaterial.Sample(uv);
    float depth = gDepth.Sample(uv);

    if (settingsUbo.debug.x == 1)
    {
        return float4(albedoData.rgb, 1.0);
    }
    if (settingsUbo.debug.x == 2)
    {
        return float4(normalData.rgb, 1.0);
    }
    if (settingsUbo.debug.x == 3)
    {
        return float4(materialData.rgb, 1.0);
    }
    if (settingsUbo.debug.x == 4)
    {
        // Most intuitive depth view: near = black, far = white.
        float linearDepth = linearizeDepth(uv, depth); // 0 near, 1 far
        return float4(linearDepth.xxx, 1.0);
    }

    float3 albedo = albedoData.rgb;
    float3 N = normalize(normalData.rgb * 2.0 - 1.0);
    float metallic = saturate(materialData.x);
    float roughness = clamp(materialData.y, 0.04, 1.0);
    float ao = saturate(materialData.z);

    float3 worldPos = reconstructWorldPos(uv, depth);
    float3 V = normalize(sceneUbo.camPos.xyz - worldPos);

    float3 F0 = lerp(float3(0.04, 0.04, 0.04), albedo, metallic);

    float3 Lo = float3(0.0, 0.0, 0.0);

    [unroll]
    for (int i = 0; i < 4; ++i)
    {
        float3 lightPos = lightUbo.lights[i].position.xyz;
        float3 lightColor = lightUbo.lights[i].color.xyz;
        float intensity = lightUbo.lights[i].color.w;

        float3 Lvec = lightPos - worldPos;
        float dist2 = max(dot(Lvec, Lvec), 0.001);
        float3 L = normalize(Lvec);
        float3 H = normalize(V + L);

        float radiance = intensity / dist2;
        float3 lightRadiance = lightColor * radiance;

        float dotNL = saturate(dot(N, L));
        float dotNV = saturate(dot(N, V));
        float dotNH = saturate(dot(N, H));
        float dotVH = saturate(dot(V, H));

        if (dotNL > 0.0 && dotNV > 0.0)
        {
            float D = D_GGX(dotNH, roughness);
            float G = G_SchlickSmith(dotNL, dotNV, roughness);
            float3 F = F_Schlick(dotVH, F0);

            float3 numerator = D * G * F;
            float denominator = max(4.0 * dotNL * dotNV, 1e-4);
            float3 specular = numerator / denominator;

            float3 kS = F;
            float3 kD = (1.0 - kS) * (1.0 - metallic);
            float3 diffuse = kD * albedo / PI;

            Lo += (diffuse + specular) * lightRadiance * dotNL;
        }
    }

    float ambientStrength = settingsUbo.params0.x;
    float exposure = settingsUbo.params0.y;
    float gamma = settingsUbo.params0.z;

    float3 ambient = albedo * ambientStrength * ao;
    float3 color = ambient + Lo;

    color = 1.0 - exp(-color * exposure);
    color = tonemapReinhard(color);
    color = pow(color, 1.0 / max(gamma, 0.01));

    return float4(color, 1.0);
}
