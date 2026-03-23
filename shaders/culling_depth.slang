struct InstanceData
{
    float4x4 model;
    float4 color;
};
[[vk::binding(1, 0)]]
StructuredBuffer<InstanceData> instanceBuffer;

struct SceneUBO
{
    float4x4 projection;
    float4x4 view;
    float3 camPos;
};
[[vk::binding(0, 0)]]
ConstantBuffer<SceneUBO> sceneUbo;

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
    output.pos = mul(sceneUbo.projection, mul(sceneUbo.view, worldPos));
    return output;
}

[shader("fragment")]
void fragMain()
{
}
