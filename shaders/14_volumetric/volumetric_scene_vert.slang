// Volumetric Lighting - Scene Vertex Shader
// Renders scene and outputs view-space position for volumetric effect

struct SceneUBO
{
    float4x4 projection;
    float4x4 view;
    float3 camPos;
    float pad0;
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

struct VSInput
{
    float3 inPosition : POSITION;
    float3 inNormal : NORMAL;
    float2 inTexCoord : TEXCOORD0;
    float4 instanceModel0 : INSTANCE_ROW_0;
    float4 instanceModel1 : INSTANCE_ROW_1;
    float4 instanceModel2 : INSTANCE_ROW_2;
    float4 instanceModel3 : INSTANCE_ROW_3;
    float4 instanceColor : COLOR0;
};

struct VSOutput
{
    float4 position : SV_Position;
    float4 viewPos : TEXCOORD0;
    float3 worldPos : TEXCOORD1;
    float3 normal : TEXCOORD2;
    float4 color : TEXCOORD3;
};

[shader("vertex")]
VSOutput vertMain(VSInput input, uint instanceID : SV_InstanceID)
{
    VSOutput output;

    InstanceData inst = instanceBuffer[instanceID];
    float4x4 modelMatrix = inst.model;

    float4 worldPos = mul(modelMatrix, float4(input.inPosition, 1.0));
    output.worldPos = worldPos.xyz;

    float4 viewPos = mul(sceneUbo.view, worldPos);
    output.viewPos = viewPos;

    output.position = mul(sceneUbo.projection, viewPos);
    output.normal = normalize((mul(modelMatrix, float4(input.inNormal, 0.0))).xyz);
    output.color = inst.color;
    output.color.rgb *= 0.5f + 0.5f * input.inTexCoord.x; // Use texcoord to prevent optimization

    return output;
}

// Non-instanced version for floor
[shader("vertex")]
VSOutput vertMainNonInstanced(VSInput input)
{
    VSOutput output;

    float4 worldPos = float4(input.inPosition, 1.0);
    output.worldPos = worldPos.xyz;

    float4 viewPos = mul(sceneUbo.view, worldPos);
    output.viewPos = viewPos;

    output.position = mul(sceneUbo.projection, viewPos);
    output.normal = input.inNormal;
    output.color = float4(0.3, 0.3, 0.35, 1.0);

    return output;
}
