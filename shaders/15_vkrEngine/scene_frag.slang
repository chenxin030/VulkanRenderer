// scene_frag.slang — Phase 3: PBR Cook-Torrance + IBL + CSM Shadows

struct PSInput
{
    float4 position : SV_Position;
    float3 worldPos : TEXCOORD0;
    float3 normal : TEXCOORD1;
    float2 texCoord : TEXCOORD2;
    float3 tangent : TEXCOORD3;
    float3 bitangent : TEXCOORD4;
    float4 shadowPos[4] : TEXCOORD5; // shadow map UVs for 4 cascades
    float viewDepth : TEXCOORD9;     // view-space depth for cascade selection
};

// Set 0: Scene-level resources
// Binding 0: Scene UBO (projection, view, camPos, light)
struct SceneUBO
{
    float4x4 projection;
    float4x4 view;
    float3 camPos;
    float pad0;
    float3 lightDir;
    float pad1;
    float3 lightColor;
    float pad2;
};
[[vk::binding(0, 0)]]
ConstantBuffer<SceneUBO> sceneUbo;

// Binding 1-3: IBL cubemaps (combined image sampler = SamplerCube in Slang)
[[vk::binding(1, 0)]]
SamplerCube samplerIrradiance;
[[vk::binding(2, 0)]]
SamplerCube prefilteredMapSampler;
[[vk::binding(3, 0)]]
Sampler2D samplerBRDFLUT;

// Binding 4: Post-processing params (exposure, gamma, lighting mode)
struct ParamsUBO
{
    float exposure;
    float gamma;
    uint lightingMode;   // 0 = Phase1 Simple, 1 = PBR+IBL
    uint enableDirLight; // 0 = off, 1 = on
};
[[vk::binding(4, 0)]]
ConstantBuffer<ParamsUBO> paramsUbo;

// Binding 5: CSM UBO (cascade VP matrices + split depths)
struct CsmUBO
{
    float4x4 cascadeViewProj[4];
    float4 cascadeSplitDepths; // x=split1, y=split2, z=split3, w=far
};
[[vk::binding(5, 0)]]
ConstantBuffer<CsmUBO> csmUbo;

// Binding 6: CSM shadow map array
[[vk::binding(6, 0)]]
Sampler2DArray shadowMapArray;

// Binding 7: Shadow filter params
struct ShadowParamsUBO
{
    int shadowFilterMode; // 0=Hard, 1=PCF, 2=PCSS
    float pcfRadiusTexels;
    float pcssLightSizeTexels;
    float shadowBiasMin;
    float2 invShadowMapSize;
    float2 padding0;
};
[[vk::binding(7, 0)]]
ConstantBuffer<ShadowParamsUBO> shadowParams;

// Set 1: Per-material resources
// Binding 0-2: PBR textures (separate image + shared sampler)
[[vk::binding(0, 1)]]
Texture2D baseColorTexture;
[[vk::binding(1, 1)]]
Texture2D normalTexture;
[[vk::binding(2, 1)]]
Texture2D metallicRoughnessTexture;
[[vk::binding(3, 1)]]
SamplerState materialSampler;

// Binding 4: Material factor UBO
struct MaterialUBO
{
    float4 baseColorFactor; // rgba
    float metallicFactor;
    float roughnessFactor;
    float normalScale;
    float occlusionStrength;
    float3 emissiveFactor;
    float alphaCutoff;
    uint flags; // bit 0: hasBaseColorTex, bit 1: hasNormalTex, bit 2: hasMetallicRoughnessTex
    uint pad0;
    uint pad1;
};
[[vk::binding(4, 1)]]
ConstantBuffer<MaterialUBO> materialUbo;

// Constants
static const float PI = 3.1415926535897932384626433832795;

// PBR BRDF functions

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

// CSM Shadow Sampling

// Select cascade index based on view-space depth
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

    // Step 1: Blocker search
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

    // Step 2: Estimate penumbra size (clamped to prevent GPU hang)
    float avgBlockerDepth = blockerDepth / float(blockerCount);
    float penumbraRatio = (referenceDepth - avgBlockerDepth) / max(avgBlockerDepth, 0.001);
    int pcfRadius = int(clamp(penumbraRatio * shadowParams.pcssLightSizeTexels, 1.0, shadowParams.pcssLightSizeTexels));

    // Step 3: PCF with adaptive radius
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

// Main CSM shadow lookup: cascade selection + shadow map sample
float csmShadowFactor(float4 shadowPos[4], float3 worldPos, float3 normal, float3 lightDir, float viewDepth)
{
    uint cascadeIdx = selectCascade(viewDepth);

    // Perspective divide to get UV + depth in shadow map space
    float4 shadowCoord = shadowPos[cascadeIdx];
    float3 projCoords = shadowCoord.xyz / shadowCoord.w;

    // Transform from [-1,1] to [0,1]
    float2 uv = projCoords.xy * 0.5 + 0.5;
    float referenceDepth = projCoords.z; // GLM_FORCE_DEPTH_ZERO_TO_ONE: already in [0,1]

    // Clamp to valid range
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0)
        return 1.0;

    // Slope-scaled bias
    float bias = max(shadowParams.shadowBiasMin * (1.0 - dot(normal, lightDir)),
                     0.0001);

    float3 sampleUV = float3(uv, float(cascadeIdx));

    // Filter mode switch
    int mode = shadowParams.shadowFilterMode;
    float shadow = 1.0;

    if (mode == 0)
    {
        shadow = sampleShadowHard(sampleUV, referenceDepth, bias);
    }
    else if (mode == 1)
    {
        shadow = sampleShadowPCF(sampleUV, referenceDepth, bias);
    }
    else
    {
        shadow = sampleShadowPCSS(sampleUV, referenceDepth, bias);
    }

    return shadow;
}

// Tone mapping

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

// Main

[shader("fragment")]
float4 fragMain(PSInput input) : SV_Target0
{
    // --- Sample textures ---
    bool hasBaseColorTex = (materialUbo.flags & 0x1u) != 0u;
    bool hasNormalTex = (materialUbo.flags & 0x2u) != 0u;
    bool hasMetallicRoughnessTex = (materialUbo.flags & 0x4u) != 0u;

    float4 baseColorSample = hasBaseColorTex
                                 ? baseColorTexture.Sample(materialSampler, input.texCoord)
                                 : float4(1.0, 1.0, 1.0, 1.0);
    float3 albedo = baseColorSample.rgb * materialUbo.baseColorFactor.rgb;

    float2 mrSample = hasMetallicRoughnessTex
                          ? metallicRoughnessTexture.Sample(materialSampler, input.texCoord).bg
                          : float2(materialUbo.metallicFactor, materialUbo.roughnessFactor);
    float metallic = mrSample.x * materialUbo.metallicFactor;
    float roughness = max(mrSample.y * materialUbo.roughnessFactor, 0.04);

    // --- Normal mapping ---
    float3 N;
    if (hasNormalTex)
    {
        float3 tangentNormal = normalTexture.Sample(materialSampler, input.texCoord).rgb * 2.0 - 1.0;
        tangentNormal.xy *= materialUbo.normalScale;
        float3 T = normalize(input.tangent);
        float3 B = normalize(input.bitangent);
        float3 Ng = normalize(input.normal);
        N = normalize(tangentNormal.x * T + tangentNormal.y * B + tangentNormal.z * Ng);
    }
    else
    {
        N = normalize(input.normal);
    }

    float3 V = normalize(sceneUbo.camPos - input.worldPos);
    float3 R = reflect(-V, N);
    float NdotV = max(dot(N, V), 0.0);

    float3 F0 = lerp(float3(0.04, 0.04, 0.04), albedo, metallic);

    // --- IBL diffuse (irradiance) ---
    float3 irradiance = samplerIrradiance.Sample(N).rgb;
    float3 diffuseIBL = irradiance * albedo;

    // --- IBL specular (prefiltered env map + BRDF LUT) ---
    float2 brdf = samplerBRDFLUT.Sample(float2(NdotV, roughness)).rg;
    float3 reflection = prefilteredReflection(R, roughness).rgb;
    float3 F = F_SchlickR(NdotV, F0, roughness);
    float3 specularIBL = reflection * (F * brdf.x + brdf.y);

    // --- Combine IBL ---
    float3 kD = (1.0 - F) * (1.0 - metallic);
    float3 ambient = kD * diffuseIBL + specularIBL;

    // --- Directional light with CSM shadow (direction from SceneUBO) ---
    float3 lightTravelDir = normalize(sceneUbo.lightDir); // direction light TRAVELS (from source)
    float3 L = -lightTravelDir;                           // direction TOWARD light source (for N·L)
    float3 lightColor = sceneUbo.lightColor;

    // Compute cascade index (for visualization) and shadow factor
    uint cascadeIdx = selectCascade(input.viewDepth);
    float shadowFactor = csmShadowFactor(input.shadowPos, input.worldPos, N, L, input.viewDepth);

    float3 H = normalize(V + L);
    float dotNL = max(dot(N, L), 0.0);
    float dotNV = NdotV;
    float dotNH = max(dot(N, H), 0.0);
    float dotVH = max(dot(V, H), 0.0);

    float3 Fd = F_Schlick(dotVH, F0);
    float D = D_GGX(dotNH, roughness);
    float G = G_SchlicksmithGGX(dotNL, dotNV, roughness);
    float3 spec = (D * G * Fd) / (4.0 * dotNL * dotNV + 0.001);

    float3 kS = Fd;
    float3 kDd = (1.0 - kS) * (1.0 - metallic);
    float3 diffuse = kDd * albedo / PI;

    float3 directLight = (diffuse + spec) * lightColor * dotNL * shadowFactor;

    // --- Combine ---
    float3 color;

    if (paramsUbo.lightingMode == 0u)
    {
        // ================================================================
        // Phase 1: Simple lighting (ambient + diffuse directional)
        // ================================================================
        float3 N_ = normalize(input.normal);
        float3 albedo_ = hasBaseColorTex
                             ? baseColorTexture.Sample(materialSampler, input.texCoord).rgb
                             : materialUbo.baseColorFactor.rgb;

        float3 lightDir_ = normalize(float3(0.3, 0.8, 0.5));
        float NdotL_ = max(dot(N_, lightDir_), 0.0);

        float3 ambient_ = float3(0.06, 0.06, 0.08) * albedo_;
        float3 diffuse_ = albedo_ * NdotL_ * float3(1.0, 0.95, 0.9);
        color = ambient_;
        if (paramsUbo.enableDirLight != 0u)
            color += diffuse_;

        // Reinhard tonemap + gamma
        color = color / (color + float3(1.0, 1.0, 1.0));
        color = pow(color, (1.0 / 2.2).xxx);
    }
    else
    {
        // ================================================================
        // Phase 2: PBR + IBL (Cook-Torrance GGX + environment lighting)
        // ================================================================
        color = ambient * 0.35; // IBL ambient for indirect fill light
        if (paramsUbo.enableDirLight != 0u)
            color += directLight;
        color += materialUbo.emissiveFactor;

        // Visualize cascades: tint each cascade a different color
        if (shadowParams.shadowFilterMode == 3)
        {
            const float3 cascadeColors[4] = {
                float3(1.0, 0.2, 0.2), // red   = cascade 0 (near)
                float3(0.2, 1.0, 0.2), // green = cascade 1
                float3(0.2, 0.2, 1.0), // blue  = cascade 2
                float3(1.0, 1.0, 0.2), // yellow= cascade 3 (far)
            };
            color = lerp(color, color * cascadeColors[cascadeIdx], 0.5);
        }

        // ---- CSM Debug Visualization ----
        // shadowFilterMode == 4: raw shadow map sample (red=near=0, blue=far=1)
        if (shadowParams.shadowFilterMode == 4)
        {
            uint dbgCascade = selectCascade(input.viewDepth);
            float4 dbgCoord = input.shadowPos[dbgCascade];
            float3 dbgProj = dbgCoord.xyz / dbgCoord.w;
            float2 dbgUV = dbgProj.xy * 0.5 + 0.5;
            float dbgDepth = dbgProj.z;
            // Sample shadow map directly
            float mapVal = shadowMapArray.SampleLevel(float3(dbgUV, float(dbgCascade)), 0).r;
            // Green = in bounds, Red = UV out of bounds, Blue = depth out of bounds
            if (dbgUV.x < 0.0 || dbgUV.x > 1.0 || dbgUV.y < 0.0 || dbgUV.y > 1.0)
                color = float3(1.0, 0.0, 0.0); // Red = UV OOB
            else if (dbgDepth < 0.0 || dbgDepth > 1.0)
                color = float3(0.0, 0.0, 1.0); // Blue = depth OOB
            else
                color = float3(mapVal, 0.0, 1.0 - mapVal); // purple=far, green+=near
        }

        // shadowFilterMode == 5: visualize shadow factor (black=shadow, white=lit)
        if (shadowParams.shadowFilterMode == 5)
        {
            color = shadowFactor.xxx;
        }

        // shadowFilterMode == 6: visualize projCoords.z (NDC depth heatmap)
        // Red=negative, Green=0-1, Blue=>1
        if (shadowParams.shadowFilterMode == 6)
        {
            uint dbgCascade = selectCascade(input.viewDepth);
            float4 dbgCoord = input.shadowPos[dbgCascade];
            float w = dbgCoord.w;
            float3 dbgProj = dbgCoord.xyz / w;
            float z = dbgProj.z;
            // Heatmap: <0 = red, 0~1 = green gradient, >1 = blue
            if (z < 0.0)
                color = float3(1.0, 0.0, 0.0) * (1.0 + z * 0.1); // red, brighter as more negative
            else if (z > 1.0)
                color = float3(0.0, 0.0, 1.0) * min(z, 3.0); // blue, brighter as >1
            else
                color = float3(0.0, z, 0.0); // green, brighter as deeper
        }

        // shadowFilterMode == 7: visualize shadowCoord.w (should be ~1 for ortho)
        if (shadowParams.shadowFilterMode == 7)
        {
            uint dbgCascade = selectCascade(input.viewDepth);
            float4 dbgCoord = input.shadowPos[dbgCascade];
            float w = dbgCoord.w;
            if (abs(w - 1.0) < 0.001)
                color = float3(0.0, 1.0, 0.0); // green = w ~ 1 (correct)
            else if (w < 1.0)
                color = float3(1.0, 1.0 - w, 0.0); // orange = w < 1
            else
                color = float3(0.0, 1.0 - 1.0 / w, 1.0); // cyan = w > 1
        }

        // Uncharted2 tonemap + gamma
        color = Uncharted2Tonemap(color * paramsUbo.exposure);
        color = color * (1.0 / Uncharted2Tonemap(float3(11.2, 11.2, 11.2)));
        color = pow(color, (1.0 / paramsUbo.gamma).xxx);
    }

    return float4(color, 1.0);
}
