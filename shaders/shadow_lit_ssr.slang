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
    float4 pointLightPosIntensity;
    float4 pointLightColor;
    float4 areaLightPosIntensity;
    float4 areaLightColor;
    float4 areaLightU;
    float4 areaLightV;
};
[[vk::binding(2, 0)]]
ConstantBuffer<ShadowUBO> shadowUbo;

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
Sampler2D shadowMap;

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

    output.pos = mul(sceneUbo.projection, mul(sceneUbo.view, float4(worldPos, 1.0)));
    output.worldPos = worldPos;
    output.worldNormal = worldNormal;
    output.shadowPos = mul(shadowUbo.lightViewProj, float4(worldPos, 1.0));
    output.color = instanceBuffer[instanceID].color;
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

    if (shadowParams.shadowFilterMode == 0)
    {
        float closestDepth = shadowMap.Sample(uv).x;
        float visibility = (currentDepth - bias) > closestDepth ? 0.0 : 1.0;
        return lerp(0.25, 1.0, visibility);
    }

    if (shadowParams.shadowFilterMode == 1)
    {
        float2 radius = shadowParams.pcfRadiusTexels * texelSize;
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

struct FSOutput
{
    float4 color : SV_Target0;
    float4 normal : SV_Target1;
};

[shader("fragment")]
FSOutput fragMain(VSOutput input)
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

    float3 pointPos = shadowUbo.pointLightPosIntensity.xyz;
    float pointIntensity = shadowUbo.pointLightPosIntensity.w;
    float3 pointColor = shadowUbo.pointLightColor.xyz;
    float3 pointVec = pointPos - input.worldPos;
    float pointDist = length(pointVec);
    float3 pointL = pointVec / max(pointDist, 1e-4);
    float pointAtt = 1.0 / (1.0 + pointDist * pointDist);
    float pointNdotl = saturate(dot(normal, pointL));
    float3 pointHalf = normalize(pointL + viewDir);
    float pointSpec = pow(saturate(dot(normal, pointHalf)), 64.0);
    float3 pointDiffuse = baseColor * pointNdotl * pointIntensity * pointAtt * pointColor;
    float3 pointSpecular = pointSpec.xxx * 0.20 * pointIntensity * pointAtt * pointColor;

    float3 areaCenter = shadowUbo.areaLightPosIntensity.xyz;
    float areaIntensity = shadowUbo.areaLightPosIntensity.w;
    float3 areaColor = shadowUbo.areaLightColor.xyz;
    float3 areaU = shadowUbo.areaLightU.xyz;
    float3 areaV = shadowUbo.areaLightV.xyz;
    float2 areaOffsets[4] = { float2(-1.0, -1.0), float2(-1.0, 1.0), float2(1.0, -1.0), float2(1.0, 1.0) };

    float3 areaSum = float3(0.0);
    for (int i = 0; i < 4; ++i)
    {
        float3 samplePos = areaCenter + areaU * areaOffsets[i].x + areaV * areaOffsets[i].y;
        float3 v = samplePos - input.worldPos;
        float dist = length(v);
        float3 L = v / max(dist, 1e-4);
        float att = 1.0 / (1.0 + dist * dist);
        float ndotl = saturate(dot(normal, L));
        float3 H = normalize(L + viewDir);
        float spec = pow(saturate(dot(normal, H)), 64.0);
        float3 diff = baseColor * ndotl;
        float3 sp = spec.xxx * 0.15;
        areaSum += (diff + sp) * att;
    }
    float3 areaLight = areaSum * (areaIntensity * 0.25) * areaColor;

    float3 color = ambient + dirShadow * (dirDiffuse + dirSpecular) + pointDiffuse + pointSpecular + areaLight;

    FSOutput outp;
    outp.color = float4(color, saturate(input.color.w));
    // 世界法线编码到 [0,1]
    outp.normal = float4(normalize(input.worldNormal) * 0.5 + 0.5, 1.0);
    return outp;
}
