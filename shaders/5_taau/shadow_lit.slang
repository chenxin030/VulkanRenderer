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

struct ShadowUBO
{
    float4x4 lightViewProj;
    float4x4 prevViewProj;
    float4 dirLightDirIntensity;
    float4 dirLightColor;
};
[[vk::binding(2, 0)]]
ConstantBuffer<ShadowUBO> shadowUbo;

struct ShadowParamsUBO
{
    int shadowFilterMode;
    float pcssLightSizeTexels;
    float shadowBiasMin;
    float2 invShadowMapSize;
    float2 padding0;
};
[[vk::binding(4, 0)]]
ConstantBuffer<ShadowParamsUBO> shadowParams;

[[vk::binding(3, 0)]]
Texture2D<float> shadowMap;
[[vk::binding(3, 0)]]
SamplerComparisonState shadowMapSampler;

struct VSInput
{
    float3 inPosition;
    float3 inNormal;
};

struct VSOutput
{
    float4 pos : SV_Position;
    float3 worldPos;
    float3 worldNormal;
    float4 shadowPos;
    float4 color;
    float4 currClipPos;
    float4 prevClipPos;
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

    float4x4 view = sceneUbo.view;
    float4x4 projection = sceneUbo.projection;

    float4 clipPos = mul(projection, mul(view, float4(worldPos, 1.0)));
    float4 prevClipPos = mul(shadowUbo.prevViewProj, float4(worldPos, 1.0));

    output.pos = clipPos;
    output.worldPos = worldPos;
    output.worldNormal = worldNormal;
    output.shadowPos = mul(shadowUbo.lightViewProj, float4(worldPos, 1.0));
    output.color = instanceBuffer[instanceID].color;
    output.currClipPos = clipPos;
    output.prevClipPos = prevClipPos;
    return output;
}

float computeShadow(float4 shadowPos, float3 normal, float3 lightDir)
{
    float3 projCoords = shadowPos.xyz / shadowPos.w;
    float2 uv = projCoords.xy * 0.5 + 0.5;
    float currentDepth = projCoords.z;

    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0)
        return 1.0;

    if (currentDepth < 0.0 || currentDepth > 1.0)
        return 1.0;

    float ndotl = saturate(dot(normal, lightDir));
    float bias = max(0.0018 * (1.0 - ndotl), shadowParams.shadowBiasMin);

    float2 texelSize = shadowParams.invShadowMapSize;

    float2 poisson[16] = {
        float2(-0.94201624, -0.39906216),
        float2(0.94558609, -0.76890725),
        float2(-0.09418410, -0.92938870),
        float2(0.34495938, 0.29387760),
        float2(-0.91588581, 0.45771432),
        float2(-0.81544232, -0.87912464),
        float2(-0.38277543, 0.27676845),
        float2(0.97484398, 0.75648379),
        float2(0.44323325, -0.97511554),
        float2(0.53742981, -0.47373420),
        float2(-0.26496911, -0.41893023),
        float2(0.79197514, 0.19090188),
        float2(-0.24188840, 0.99706507),
        float2(-0.81409955, 0.91437590),
        float2(0.19984126, 0.78641367),
        float2(0.14383161, -0.14100790)
    };

    float2 searchRadius = shadowParams.pcssLightSizeTexels * texelSize;
    float blockerSum = 0.0;
    float blockerCount = 0.0;
    for (int i = 0; i < 16; ++i)
    {
        float2 suv = uv + poisson[i] * searchRadius;
        float d = shadowMap.Sample(suv).x;
        if (d < (currentDepth - bias))
        {
            blockerSum += d;
            blockerCount += 1.0;
        }
    }
    if (blockerCount < 0.5)
    {
        return 1.0;
    }

    float avgBlockerDepth = blockerSum / blockerCount;
    float penumbra = (currentDepth - avgBlockerDepth) / max(avgBlockerDepth, 1e-4);
    float filterRadiusTexels = clamp(penumbra * shadowParams.pcssLightSizeTexels, 1.0, 80.0);

    float2 radius = filterRadiusTexels * texelSize;
    float sum = 0.0;
    for (int i = 0; i < 16; ++i)
    {
        float2 suv = uv + poisson[i] * radius;
        float closestDepth = shadowMap.Sample(suv).x;
        sum += (currentDepth - bias) > closestDepth ? 0.0 : 1.0;
    }
    float visibility = sum / 16.0;
    return lerp(0.25, 1.0, visibility);
}

[shader("fragment")]
void fragMain(VSOutput input, out float4 color : SV_TARGET0, out float2 velocity : SV_TARGET1)
{
    float3 baseColor = input.color.xyz;
    float3 normal = normalize(input.worldNormal);

    float3 viewDir = normalize(sceneUbo.camPos - input.worldPos);

    float3 ambient = baseColor * 0.06;

    float3 dirL = normalize(-shadowUbo.dirLightDirIntensity.xyz);
    float dirIntensity = shadowUbo.dirLightDirIntensity.w;
    float3 dirColor = shadowUbo.dirLightColor.xyz;
    float dirNdotl = saturate(dot(normal, dirL));
    float3 dirHalf = normalize(dirL + viewDir);
    float dirSpec = pow(saturate(dot(normal, dirHalf)), 64.0);
    float dirShadow = computeShadow(input.shadowPos, normal, dirL);
    float3 dirDiffuse = baseColor * dirNdotl * dirIntensity * dirColor;
    float3 dirSpecular = dirSpec.xxx * 0.25 * dirIntensity * dirColor;

    color = float4(ambient + dirShadow * (dirDiffuse + dirSpecular), 1.0);

    float2 currNDC = input.currClipPos.xy / input.currClipPos.w;
    float2 prevNDC = input.prevClipPos.xy / input.prevClipPos.w;
    velocity = currNDC - prevNDC;
}

