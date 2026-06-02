// deferred_lighting.slang — Phase 4: Deferred lighting pass for vkrEngine
// Full-screen triangle, reads GBuffer + SSAO + SSR + IBL + CSM
// Outputs final lit color to swapchain

struct VSOutput
{
    float4 Pos : SV_POSITION;
    float2 uv : TEXCOORD0;
};

// Set 0: Scene-level resources
struct SceneUBO
{
    float4x4 projection;
    float4x4 view;
    float4x4 invProjection;
    float4x4 invView;
    float4 camPos;
    float4 lightDir;
    float4 lightColor;
};
[[vk::binding(0, 0)]]
ConstantBuffer<SceneUBO> sceneUbo;

// IBL
[[vk::binding(1, 0)]]
SamplerCube samplerIrradiance;
[[vk::binding(2, 0)]]
SamplerCube prefilteredMapSampler;
[[vk::binding(3, 0)]]
Sampler2D samplerBRDFLUT;

// Params
struct ParamsUBO
{
    float exposure;
    float gamma;
    uint lightingMode;
    uint enableDirLight;
};
[[vk::binding(4, 0)]]
ConstantBuffer<ParamsUBO> paramsUbo;

// CSM
struct CsmUBO
{
    float4x4 cascadeViewProj[4];
    float4 cascadeSplitDepths;
};
[[vk::binding(5, 0)]]
ConstantBuffer<CsmUBO> csmUbo;

// Shadow
[[vk::binding(6, 0)]]
Sampler2DArray shadowMapArray;

struct ShadowParamsUBO
{
    int shadowFilterMode;
    float pcfRadiusTexels;
    float pcssLightSizeTexels;
    float shadowBiasMin;
    float2 invShadowMapSize;
    float2 padding0;
};
[[vk::binding(7, 0)]]
ConstantBuffer<ShadowParamsUBO> shadowParams;

// GBuffer inputs (Set 1)
[[vk::binding(0, 1)]]
Sampler2D<float4> gAlbedo;
[[vk::binding(1, 1)]]
Sampler2D<float4> gNormalRoughness;
[[vk::binding(2, 1)]]
Sampler2D<float4> gPbr;
[[vk::binding(3, 1)]]
Sampler2D<float> gDepth;

// SSAO
[[vk::binding(4, 1)]]
Sampler2D<float> gSSAO;

// SSR
[[vk::binding(5, 1)]]
Sampler2D<float4> gSSR;

// Deferred settings (Set 1, Binding 6)
struct DeferredSettingsUBO
{
    float ssaoEnabled;
    float ssrEnabled;
    float debugView;
    float pad0;
};
[[vk::binding(6, 1)]]
ConstantBuffer<DeferredSettingsUBO> deferredSettings;

static const float PI = 3.14159265358979323846;

// ====================================================================
// PBR BRDF
// ====================================================================

float D_GGX(float dotNH, float roughness)
{
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    float denom = dotNH * dotNH * (alpha2 - 1.0) + 1.0;
    return alpha2 / (PI * denom * denom);
}

float G_SchlicksmithGGX(float dotNL, float dotNV, float roughness)
{
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    float GL = dotNL / (dotNL * (1.0 - k) + k);
    float GV = dotNV / (dotNV * (1.0 - k) + k);
    return GL * GV;
}

float3 F_Schlick(float cosTheta, float3 F0)
{
    return F0 + (1.0 - F0) * pow(max(1.0 - cosTheta, 0.0), 5.0);
}

float3 F_SchlickR(float cosTheta, float3 F0, float roughness)
{
    return F0 + (max((1.0 - roughness).xxx, F0) - F0) * pow(max(1.0 - cosTheta, 0.0), 5.0);
}

float3 prefilteredReflection(float3 R, float roughness)
{
    const float MAX_REFLECTION_LOD = 9.0;
    float lod = roughness * MAX_REFLECTION_LOD;
    float lodf = floor(lod);
    float lodc = ceil(lod);
    float3 a = prefilteredMapSampler.SampleLevel(R, lodf).rgb;
    float3 b = prefilteredMapSampler.SampleLevel(R, lodc).rgb;
    return lerp(a, b, lod - lodf);
}

// ====================================================================
// Position reconstruction
// ====================================================================

float3 reconstructWorldPos(float2 uv, float depth)
{
    float4 clip = float4(uv * 2.0 - 1.0, depth, 1.0);
    float4 viewPos = mul(sceneUbo.invProjection, clip);
    viewPos /= max(viewPos.w, 1e-6);
    float4 worldPos = mul(sceneUbo.invView, viewPos);
    return worldPos.xyz;
}

// ====================================================================
// CSM Shadow
// ====================================================================

uint selectCascade(float viewDepth)
{
    uint cascade = 0u;
    if (viewDepth > csmUbo.cascadeSplitDepths.x)
        cascade = 1u;
    if (viewDepth > csmUbo.cascadeSplitDepths.y)
        cascade = 2u;
    if (viewDepth > csmUbo.cascadeSplitDepths.z)
        cascade = 3u;
    return cascade;
}

float sampleShadowHard(float3 uv, float referenceDepth, float bias)
{
    float mapDepth = shadowMapArray.SampleLevel(uv, 0).r;
    return (referenceDepth - bias <= mapDepth) ? 1.0 : 0.0;
}

float sampleShadowPCF(float3 uv, float referenceDepth, float bias)
{
    float shadow = 0.0;
    int radius = int(shadowParams.pcfRadiusTexels);
    float2 texelSize = shadowParams.invShadowMapSize;

    for (int x = -radius; x <= radius; x++)
    {
        for (int y = -radius; y <= radius; y++)
        {
            float3 sampleUV = uv + float3(float2(x, y) * texelSize, 0.0);
            float mapDepth = shadowMapArray.SampleLevel(sampleUV, 0).r;
            shadow += (referenceDepth - bias <= mapDepth) ? 1.0 : 0.0;
        }
    }
    float sampleCount = (float(radius) * 2.0 + 1.0) * (float(radius) * 2.0 + 1.0);
    return shadow / sampleCount;
}

float sampleShadowPCSS(float3 uv, float referenceDepth, float bias)
{
    float2 texelSize = shadowParams.invShadowMapSize;

    // Blocker search
    float blockerDepth = 0.0;
    int blockerCount = 0;
    int searchRadius = int(shadowParams.pcssLightSizeTexels * 0.5);

    for (int x = -searchRadius; x <= searchRadius; x++)
    {
        for (int y = -searchRadius; y <= searchRadius; y++)
        {
            float3 sampleUV = uv + float3(float2(x, y) * texelSize, 0.0);
            float mapDepth = shadowMapArray.SampleLevel(sampleUV, 0).r;
            if (referenceDepth - bias > mapDepth)
            {
                blockerDepth += mapDepth;
                blockerCount++;
            }
        }
    }

    if (blockerCount == 0)
        return 1.0;

    float avgBlockerDepth = blockerDepth / float(blockerCount);
    float penumbraRatio = (referenceDepth - avgBlockerDepth) / max(avgBlockerDepth, 0.001);
    int pcfRadius = int(clamp(penumbraRatio * shadowParams.pcssLightSizeTexels, 1.0, shadowParams.pcssLightSizeTexels));

    float shadow = 0.0;
    for (int x = -pcfRadius; x <= pcfRadius; x++)
    {
        for (int y = -pcfRadius; y <= pcfRadius; y++)
        {
            float3 sampleUV = uv + float3(float2(x, y) * texelSize, 0.0);
            float mapDepth = shadowMapArray.SampleLevel(sampleUV, 0).r;
            shadow += (referenceDepth - bias <= mapDepth) ? 1.0 : 0.0;
        }
    }
    float sampleCount = (float(pcfRadius) * 2.0 + 1.0) * (float(pcfRadius) * 2.0 + 1.0);
    return shadow / sampleCount;
}

// Simplified CSM for deferred: reconstruct shadow coordinate from worldPos
float csmShadowFactorDeferred(float3 worldPos, float3 N, float3 L)
{
    // Compute view-space depth for cascade selection
    float4 viewPos4 = mul(sceneUbo.view, float4(worldPos, 1.0));
    float viewDepth = -viewPos4.z;

    uint cascadeIdx = selectCascade(viewDepth);

    // Use cascade VP matrix
    float4 shadowCoord = mul(csmUbo.cascadeViewProj[cascadeIdx], float4(worldPos, 1.0));
    float3 projCoords = shadowCoord.xyz / shadowCoord.w;

    float2 uv = projCoords.xy * 0.5 + 0.5;
    float referenceDepth = projCoords.z;

    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0)
        return 1.0;

    float bias = max(shadowParams.shadowBiasMin * (1.0 - dot(N, L)), 0.0001);
    float3 sampleUV = float3(uv, float(cascadeIdx));

    int mode = shadowParams.shadowFilterMode;
    if (mode == 0)
        return sampleShadowHard(sampleUV, referenceDepth, bias);
    else if (mode == 1)
        return sampleShadowPCF(sampleUV, referenceDepth, bias);
    else
        return sampleShadowPCSS(sampleUV, referenceDepth, bias);
}

// ====================================================================
// Tone mapping
// ====================================================================

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

// ====================================================================
// Entry
// ====================================================================

[shader("vertex")]
VSOutput vertMain(uint vertexID: SV_VertexID)
{
    VSOutput output;

    float2 pos;
    if (vertexID == 0u)
        pos = float2(-1.0, -1.0);
    else if (vertexID == 1u)
        pos = float2(3.0, -1.0);
    else
        pos = float2(-1.0, 3.0);

    output.Pos = float4(pos, 0.0, 1.0);
    output.uv = pos * 0.5 + 0.5;
    return output;
}

[shader("fragment")]
float4 fragMain(VSOutput input) : SV_Target0
{
    float2 uv = input.uv;

    // --- Read GBuffer ---
    float4 albedoData = gAlbedo.Sample(uv);
    float4 normalRoughness = gNormalRoughness.Sample(uv);
    float4 pbrData = gPbr.Sample(uv);
    float depth = gDepth.Sample(uv);

    // --- Debug views ---
    int debugView = int(deferredSettings.debugView);
    if (debugView == 1)
        return float4(albedoData.rgb, 1.0);
    if (debugView == 2)
        return float4(normalRoughness.rgb, 1.0);
    if (debugView == 3)
        return float4(pbrData.rgb, 1.0);
    if (debugView == 4)
    {
        // Linearize depth for better visualization
        float3 worldPos = reconstructWorldPos(uv, depth);
        float viewDist = distance(sceneUbo.camPos.xyz, worldPos);
        float linearDepth = saturate(viewDist / 100.0); // 0=near, 1=100+ units
        return float4(linearDepth.xxx, 1.0);
    }

    // --- Unpack ---
    float3 albedo = albedoData.rgb;
    float3 N = normalize(normalRoughness.rgb * 2.0 - 1.0);
    float roughness = clamp(normalRoughness.a, 0.04, 1.0);
    float metallic = saturate(pbrData.r);
    float ao = saturate(pbrData.g);

    float3 worldPos = reconstructWorldPos(uv, depth);
    float3 V = normalize(sceneUbo.camPos.xyz - worldPos);
    float3 R = reflect(-V, N);
    float NdotV = max(dot(N, V), 0.0);

    float3 F0 = lerp(float3(0.04, 0.04, 0.04), albedo, metallic);

    // --- IBL ---
    float3 irradiance = samplerIrradiance.Sample(N).rgb;
    float3 diffuseIBL = irradiance * albedo;

    float2 brdf = samplerBRDFLUT.Sample(float2(NdotV, roughness)).rg;
    float3 reflection = prefilteredReflection(R, roughness).rgb;
    float3 F_ibl = F_SchlickR(NdotV, F0, roughness);
    float3 specularIBL = reflection * (F_ibl * brdf.x + brdf.y);

    float3 kD_ibl = (1.0 - F_ibl) * (1.0 - metallic);
    float3 ambient = kD_ibl * diffuseIBL + specularIBL;

    // --- Directional light with CSM ---
    float3 lightTravelDir = normalize(sceneUbo.lightDir.xyz);
    float3 L = -lightTravelDir;
    float3 lightColor = sceneUbo.lightColor.xyz;

    float3 H = normalize(V + L);
    float dotNL = max(dot(N, L), 0.0);
    float dotNH = max(dot(N, H), 0.0);
    float dotVH = max(dot(V, H), 0.0);

    float3 Fd = F_Schlick(dotVH, F0);
    float D = D_GGX(dotNH, roughness);
    float G = G_SchlicksmithGGX(dotNL, NdotV, roughness);
    float3 spec = (D * G * Fd) / (4.0 * dotNL * NdotV + 0.001);

    float3 kS = Fd;
    float3 kDd = (1.0 - kS) * (1.0 - metallic);
    float3 diffuse = kDd * albedo / PI;

    float shadowFactor = csmShadowFactorDeferred(worldPos, N, L);
    float3 directLight = (diffuse + spec) * lightColor * dotNL * shadowFactor;

    // --- SSAO ---
    float ssaoFactor = 1.0;
    if (deferredSettings.ssaoEnabled > 0.5)
        ssaoFactor = gSSAO.Sample(uv).r;

    // --- SSR ---
    float3 ssrColor = float3(0.0, 0.0, 0.0);
    if (deferredSettings.ssrEnabled > 0.5)
        ssrColor = gSSR.Sample(uv).rgb;

    // --- Combine ---
    float3 color = ambient * 0.35; // IBL ambient fill
    if (paramsUbo.enableDirLight != 0u)
        color += directLight * lerp(0.5, 1.0, ssaoFactor); // Apply SSAO to direct light
    color += ssrColor * F_ibl;                             // Modulate SSR by Fresnel

    // Apply AO to ambient too
    color *= lerp(0.3, 1.0, ssaoFactor * ao);

    // --- Tone map ---
    float exposure = paramsUbo.exposure;
    float gamma = paramsUbo.gamma;

    float3 mapped = Uncharted2Tonemap(color * exposure);
    float3 whiteScale = 1.0 / Uncharted2Tonemap(float3(11.2, 11.2, 11.2));
    color = mapped * whiteScale;
    color = pow(color, (1.0 / gamma).xxx);

    return float4(color, 1.0);
}
