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

[[vk::binding(2, 0)]]
StructuredBuffer<uint> visibleIndices;

struct VSInput
{
    float3 inPosition;
    float3 inNormal;
};

struct VSOutput
{
    float4 pos : SV_Position;
    float3 worldNormal;
    float4 color;
};

[shader("vertex")]
VSOutput vertMain(VSInput input, uint instanceID : SV_InstanceID)
{
    VSOutput output;
    uint visibleIndex = visibleIndices[instanceID];
    float4x4 model = instanceBuffer[visibleIndex].model;
    float4 worldPos = mul(model, float4(input.inPosition, 1.0));
    output.pos = mul(sceneUbo.projection, mul(sceneUbo.view, worldPos));

    float3x3 normalMat = (float3x3)model;
    output.worldNormal = normalize(mul(normalMat, input.inNormal));
    output.color = instanceBuffer[visibleIndex].color;
    return output;
}

[shader("fragment")]
float4 fragMain(VSOutput input) : SV_TARGET
{
    float3 baseColor = input.color.xyz;
    float3 lightDir = normalize(float3(0.6, -1.0, 0.4));
    float ndotl = saturate(dot(normalize(input.worldNormal), -lightDir));
    float3 color = baseColor * (0.2 + 0.8 * ndotl);
    return float4(color, 1.0);
}
