struct SceneUBO
{
    float4x4 projection;
    float4x4 view;
    float4x4 invProjection;
    float4x4 invView;
    float4 camPos;
};
[[vk::binding(0, 0)]]
ConstantBuffer<SceneUBO> sceneUbo;

struct DeferredInstanceData
{
    float4x4 model;
    float4 baseColor;
    float4 material; // x: metallic, y: roughness, z: ao, w: unused
};
[[vk::binding(1, 0)]]
StructuredBuffer<DeferredInstanceData> instanceData;

struct VSInput
{
    float3 Pos;
    float3 Normal;
    float2 texCoord; // keep vertex layout aligned with C++ Vertex stride
};

struct VSOutput
{
    float4 Pos : SV_POSITION;
    float3 WorldPos;
    float3 Normal;
    uint InstanceIndex : SV_InstanceID;
};

struct GBufferOutput
{
    float4 outAlbedo : SV_Target0;
    float4 outNormal : SV_Target1;
    float4 outMaterial : SV_Target2;
};

[[shader("vertex")]]
VSOutput vertMain(VSInput input, uint instanceID : SV_InstanceID)
{
    VSOutput output;
    DeferredInstanceData data = instanceData[instanceID];

    float4 worldPos = mul(data.model, float4(input.Pos, 1.0));
    output.WorldPos = worldPos.xyz;

    float3x3 normalMat = (float3x3)data.model;
    output.Normal = normalize(mul(normalMat, input.Normal));

    output.Pos = mul(sceneUbo.projection, mul(sceneUbo.view, worldPos));
    output.InstanceIndex = instanceID;
    return output;
}

[[shader("fragment")]]
GBufferOutput fragMain(VSOutput input)
{
    DeferredInstanceData data = instanceData[input.InstanceIndex];

    GBufferOutput outp;
    outp.outAlbedo = data.baseColor;

    float3 N = normalize(input.Normal);
    outp.outNormal = float4(N * 0.5 + 0.5, 1.0);

    outp.outMaterial = data.material;
    return outp;
}
