// scene_vert.slang — Phase 1: Simple forward rendering vertex shader

struct VSOutput
{
    float4 position : SV_Position;
    float3 worldPos : TEXCOORD0;
    float3 normal : TEXCOORD1;
    float2 texCoord : TEXCOORD2;
};

// Set 0, Binding 0: Scene UBO
struct SceneUBO
{
    float4x4 projection;
    float4x4 view;
    float3 camPos;
    float pad0;
};

[[vk::binding(0, 0)]]
cbuffer SceneUBO_ : register(b0)
{
    SceneUBO sceneUBO;
}

// Push constant: model matrix (mat4 = 64 bytes)
struct PushBlock
{
    float4x4 modelMatrix;
};
[[vk::push_constant]]
PushBlock push;

[shader("vertex")]
VSOutput vertMain(
    float3 pos: POSITION,
    float3 normal: NORMAL,
    float2 texCoord: TEXCOORD0,
    float4 tangent: TANGENT)
{
    VSOutput output;

    float4 worldPos = mul(push.modelMatrix, float4(pos, 1.0));
    output.position = mul(sceneUBO.projection, mul(sceneUBO.view, worldPos));
    output.worldPos = worldPos.xyz;
    output.normal = normalize(mul((float3x3)push.modelMatrix, normal));
    output.texCoord = texCoord;

    return output;
}
