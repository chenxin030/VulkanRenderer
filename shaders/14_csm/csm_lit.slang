// CSM Lit Pass Shader
// Samples the CSM shadow map array with cascade selection and PCF/PCSS filtering.

struct SceneUBO
{
    float4x4 projection;
    float4x4 view;
    float3 camPos;
};
[[vk::binding(0, 0)]]
ConstantBuffer<SceneUBO> sceneUbo;

struct InstanceData
{
    float4x4 model;
    float4 color;
};
[[vk::binding(1, 0)]]
StructuredBuffer<InstanceData> instanceBuffer;

struct CsmUBO
{
    float4x4 cascadeViewProj[4];
    float4 cascadeSplitDepths; // x=split1, y=split2, z=split3, w=far
};
[[vk::binding(2, 0)]]
ConstantBuffer<CsmUBO> csmUbo;

struct ShadowParamsUBO
{
    int shadowFilterMode;
    float pcfRadiusTexels;
    float pcssLightSizeTexels;
    float shadowBiasMin;
    float2 invShadowMapSize;
    float2 padding0;
};
[[vk::binding(4, 0)]]
ConstantBuffer<ShadowParamsUBO> shadowParams;

[[vk::binding(3, 0)]]
Sampler2DArray shadowMapArray;

// ---- Per-instance push constant (set from C++ side) ----
// Not used in lit pass, but declared for pipeline layout compatibility.
// The lit pipeline has no push constants; depth pipeline does.
// We use the same descriptor set layout but different pipeline layouts.

struct VSInput
{
    float3 inPosition : POSITION;
    float3 inNormal : NORMAL;
};

struct VSOutput
{
    float4 pos : SV_Position;
    float3 worldPos : TEXCOORD0;
    float3 worldNormal : TEXCOORD1;
    float4 shadowPos[4] : TEXCOORD2; // one per cascade
    float4 color : COLOR0;
    float viewDepth : TEXCOORD6;
};

[shader("vertex")]
VSOutput vertMain(VSInput input, uint instanceID: SV_InstanceID)
{
    VSOutput output;

    float4x4 model = instanceBuffer[instanceID].model;
    float4 worldPos4 = mul(model, float4(input.inPosition, 1.0));
    float3 worldPos = worldPos4.xyz;

    float3x3 normalMat = (float3x3)model;
    float3 worldNormal = normalize(mul(normalMat, input.inNormal));

    float4 viewPos = mul(sceneUbo.view, float4(worldPos, 1.0));

    output.pos = mul(sceneUbo.projection, viewPos);
    output.worldPos = worldPos;
    output.worldNormal = worldNormal;
    output.color = instanceBuffer[instanceID].color;

    // Compute shadow positions for all cascades
    [unroll]
    for (int i = 0; i < 4; i++)
    {
        output.shadowPos[i] = mul(csmUbo.cascadeViewProj[i], float4(worldPos, 1.0));
    }

    // View-space depth (positive) for cascade selection
    output.viewDepth = -viewPos.z;

    return output;
}

// ---- Shadow sampling helpers ----

float sampleShadowHard(float3 uv, float referenceDepth, float bias)
{
    float mapDepth = shadowMapArray.SampleLevel(uv, 0).r;
    return (referenceDepth - bias <= mapDepth) ? 1.0 : 0.0;
}

float sampleShadowPCF(float3 uv, float referenceDepth, float bias, float2 texelSize)
{
    float shadow = 0.0;
    int radius = int(shadowParams.pcfRadiusTexels);
    float filterSize = float(radius);

    for (int x = -radius; x <= radius; x++)
    {
        for (int y = -radius; y <= radius; y++)
        {
            float3 sampleUV = uv + float3(float2(x, y) * texelSize, 0.0);
            float mapDepth = shadowMapArray.SampleLevel(sampleUV, 0).r;
            shadow += (referenceDepth - bias <= mapDepth) ? 1.0 : 0.0;
        }
    }
    float sampleCount = (filterSize * 2.0 + 1.0) * (filterSize * 2.0 + 1.0);
    return shadow / sampleCount;
}

// PCSS: estimate blocker depth to determine penumbra size, then PCF with adaptive radius
float sampleShadowPCSS(float3 uv, float referenceDepth, float bias, float2 texelSize)
{
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

    float avgBlockerDepth = blockerDepth / float(blockerCount);

    // Step 2: Penumbra estimation
    float penumbraSize = (referenceDepth - avgBlockerDepth) / avgBlockerDepth;
    penumbraSize *= shadowParams.pcssLightSizeTexels;
    int pcfRadius = int(clamp(penumbraSize, 1.0, shadowParams.pcssLightSizeTexels));

    // Step 3: PCF with penumbra radius
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

float computeShadow(float3 worldPos, float3 normal, float3 lightDir, float viewDepth)
{
    // ---- Cascade Selection ----
    // Compare view-space depth against cascade split depths
    uint cascadeIdx = 0u;
    [unroll]
    for (int i = 0; i < 3; i++)
    {
        if (viewDepth > csmUbo.cascadeSplitDepths[i])
            cascadeIdx = uint(i + 1);
    }

    float4 shadowPos = float4(0, 0, 0, 0);

    // Manual unroll for array indexing compatibility
    if (cascadeIdx == 0u)
        shadowPos = shadowPos[0];
    else if (cascadeIdx == 1u)
        shadowPos = shadowPos[1];
    else if (cascadeIdx == 2u)
        shadowPos = shadowPos[2];
    else
        shadowPos = shadowPos[3];

    // Perspective divide
    float3 projCoords = shadowPos.xyz / shadowPos.w;
    float2 uv = projCoords.xy * 0.5 + 0.5;
    float currentDepth = projCoords.z;

    // Out-of-bounds check
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0)
        return 1.0;
    if (currentDepth < 0.0 || currentDepth > 1.0)
        return 1.0;

    // Bias
    float ndotl = saturate(dot(normal, lightDir));
    float bias = max(0.0018 * (1.0 - ndotl), shadowParams.shadowBiasMin);

    // Build 3D UV for texture array: (u, v, layer)
    float3 uv3 = float3(uv, float(cascadeIdx));

    // Filter selection
    if (shadowParams.shadowFilterMode == 0)
        return sampleShadowHard(uv3, currentDepth, bias);
    else if (shadowParams.shadowFilterMode == 1)
        return sampleShadowPCF(uv3, currentDepth, bias, shadowParams.invShadowMapSize);
    else
        return sampleShadowPCSS(uv3, currentDepth, bias, shadowParams.invShadowMapSize);
}

// ---- Cascade visualization colors ----
float3 getCascadeColor(uint cascadeIdx)
{
    if (cascadeIdx == 0u)
        return float3(1.0, 0.2, 0.2); // Red
    if (cascadeIdx == 1u)
        return float3(0.2, 1.0, 0.2); // Green
    if (cascadeIdx == 2u)
        return float3(0.2, 0.4, 1.0); // Blue
    return float3(1.0, 0.9, 0.2);     // Yellow
}

[shader("fragment")]
float4 fragMain(VSOutput input) : SV_Target
{
    float3 normal = normalize(input.worldNormal);

    // Directional light
    // Light direction: from shadow setup (rotating around origin)
    float3 lightDir = normalize(float3(0.6, -1.0, 0.6)); // approximate — matches C++

    // Ambient + diffuse
    float3 ambient = float3(0.05, 0.05, 0.08);
    float ndotl = saturate(dot(normal, lightDir));
    float3 diffuse = input.color.rgb * ndotl * 0.5;

    // Shadow
    float shadow = computeShadow(input.worldPos, normal, lightDir, input.viewDepth);

    // Determine cascade for visualization
    uint cascadeIdx = 0u;
    [unroll]
    for (int i = 0; i < 3; i++)
    {
        if (input.viewDepth > csmUbo.cascadeSplitDepths[i])
            cascadeIdx = uint(i + 1);
    }

    // Cascade visualization overlay (controlled by uniform — for now always blend)
    // We'll use a simple approach: if shadow < 0.3 it's in shadow (dark).
    // Cascade color overlay shows which cascade was used.

    // Visualize cascade with subtle tint
    float3 cascadeOverlay = float3(0.0);
    // We can't access the visualizeCascades uniform directly from C++ UBO,
    // so we use a simple toggle: if shadowFilterMode > 2 as a hack.
    // Better: just always show a subtle tint on shadow edges.

    float3 litColor = ambient + diffuse * shadow * 0.5;

    // Subtle cascade color overlay in shadow regions
    float3 cascadeTint = getCascadeColor(cascadeIdx) * 0.08;
    float3 result = litColor + cascadeTint;

    // Darken fully shadowed areas
    result = lerp(result, result * 0.25, (1.0 - shadow) * 0.7);

    return float4(result, 1.0);
}
