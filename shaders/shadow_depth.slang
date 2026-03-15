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

struct VSInput
{
    float3 inPosition;
};

struct VSOutput
{
    float4 pos : SV_Position;
};

[shader("vertex")]
VSOutput vertMain(VSInput input, uint instanceID : SV_InstanceID)
{
    VSOutput output;
    float4x4 model = instanceBuffer[instanceID].model;
    float4 worldPos = mul(model, float4(input.inPosition, 1.0));
    output.pos = mul(shadowUbo.lightViewProj, worldPos);
    return output;
}

[shader("fragment")]
void fragMain()
{
}
