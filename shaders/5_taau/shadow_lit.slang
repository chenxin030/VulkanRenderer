struct SceneUBO
{
    float4x4 projection;
    float4x4 view;
    float3 camPos;
};
[[vk::binding(1, 0)]]
ConstantBuffer<SceneUBO> sceneUbo;

struct InstanceData
{
    float4x4 model;
    float4 color;
};
[[vk::binding(0, 0)]]
StructuredBuffer<InstanceData> instanceBuffer;

struct PrevVPBuffer
{
    float4x4 prevViewProj;
};
[[vk::binding(2, 0)]]
ConstantBuffer<PrevVPBuffer> prevVpBuffer;

struct VSInput
{
    float3 inPosition;
};

struct VSOutput
{
    float4 pos : SV_Position;
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

    float4x4 view = sceneUbo.view;
    float4x4 projection = sceneUbo.projection;

    float4 currClipPos = mul(projection, mul(view, float4(worldPos, 1.0)));
    float4 prevClipPos = mul(prevVpBuffer.prevViewProj, float4(worldPos, 1.0));

    output.pos = currClipPos;
    output.color = instanceBuffer[instanceID].color;
    output.currClipPos = currClipPos;
    output.prevClipPos = prevClipPos;
    return output;
}

[shader("fragment")]
void fragMain(VSOutput input, out float4 color : SV_TARGET0, out float2 velocity : SV_TARGET1)
{
    color = input.color;

    float2 currNDC = input.currClipPos.xy / input.currClipPos.w;
    float2 prevNDC = input.prevClipPos.xy / input.prevClipPos.w;
    velocity = currNDC - prevNDC;
}
