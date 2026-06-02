// csm_depth.slang — Phase 3: CSM depth pass shader for vkrEngine
// Renders Sponza geometry (VkrVertex) into shadow map array cascade layers.
// Push constant: model matrix (mat4) + cascade index (uint).

struct CsmUBO
{
    float4x4 cascadeViewProj[4];
    float4 cascadeSplitDepths; // x=split1, y=split2, z=split3, w=far
};
[[vk::binding(0, 0)]]
ConstantBuffer<CsmUBO> csmUbo;

// Push constant: model matrix (mat4) + cascade index (uint)
struct PushBlock
{
    float4x4 modelMatrix;
    uint cascadeIndex;
};
[[vk::push_constant]]
PushBlock push;

struct VSInput
{
    float3 inPosition : POSITION;
};

struct VSOutput
{
    float4 pos : SV_Position;
};

[shader("vertex")]
VSOutput vertMain(VSInput input)
{
    VSOutput output;

    float4 worldPos = mul(push.modelMatrix, float4(input.inPosition, 1.0));
    output.pos = mul(csmUbo.cascadeViewProj[push.cascadeIndex], worldPos);

    return output;
}

[shader("fragment")]
void fragMain()
{
    // Depth is written automatically; no color output needed.
}
