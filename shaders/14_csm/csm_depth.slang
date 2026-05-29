// CSM Depth Pass Shader
// Renders scene geometry into the specified cascade layer of the shadow map array.
// cascadeIndex is passed via push constant.

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
    float4 cascadeSplitDepths;
};
[[vk::binding(2, 0)]]
ConstantBuffer<CsmUBO> csmUbo;

[[vk::push_constant]]
uint cascadeIndex;

struct VSInput
{
    float3 inPosition : POSITION;
};

struct VSOutput
{
    float4 pos : SV_Position;
};

[shader("vertex")]
VSOutput vertMain(VSInput input, uint instanceID: SV_InstanceID)
{
    VSOutput output;

    float4x4 model = instanceBuffer[instanceID].model;
    float4 worldPos = mul(model, float4(input.inPosition, 1.0));
    output.pos = mul(csmUbo.cascadeViewProj[cascadeIndex], worldPos);

    return output;
}

[shader("fragment")]
void fragMain()
{
    // Depth is written automatically; no color output needed.
}
