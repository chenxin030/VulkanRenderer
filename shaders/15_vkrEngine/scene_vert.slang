// scene_vert.slang — Phase 3: PBR + IBL + CSM vertex shader

struct VSOutput
{
    float4 position : SV_Position;
    float3 worldPos : TEXCOORD0;
    float3 normal : TEXCOORD1;
    float2 texCoord : TEXCOORD2;
    float3 tangent : TEXCOORD3;
    float3 bitangent : TEXCOORD4;
    float4 shadowPos[4] : TEXCOORD5; // shadow map UVs for 4 cascades
    float viewDepth : TEXCOORD9;     // view-space depth for cascade selection
};

// Set 0, Binding 0: Scene UBO
struct SceneUBO
{
    float4x4 projection;
    float4x4 view;
    float3 camPos;
    float pad0;
    float3 lightDir;
    float pad1;
    float3 lightColor;
    float pad2;
};

[[vk::binding(0, 0)]]
cbuffer SceneUBO_ : register(b0)
{
    SceneUBO sceneUBO;
}

// Set 0, Binding 5: CSM UBO (cascade view-projection matrices)
struct CsmUBO
{
    float4x4 cascadeViewProj[4];
    float4 cascadeSplitDepths; // x=split1, y=split2, z=split3, w=far
};
[[vk::binding(5, 0)]]
ConstantBuffer<CsmUBO> csmUbo;

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
    float4 viewPos = mul(sceneUBO.view, worldPos);
    output.position = mul(sceneUBO.projection, viewPos);
    output.worldPos = worldPos.xyz;

    float3x3 normalMatrix = (float3x3)push.modelMatrix;
    output.normal = normalize(mul(normalMatrix, normal));
    output.tangent = normalize(mul(normalMatrix, tangent.xyz));
    output.bitangent = cross(output.normal, output.tangent) * tangent.w;
    output.texCoord = texCoord;

    // Compute shadow positions for all 4 cascades
    [unroll]
    for (int i = 0; i < 4; i++)
    {
        output.shadowPos[i] = mul(csmUbo.cascadeViewProj[i], worldPos);
    }

    // View-space depth (positive) for cascade selection
    output.viewDepth = -viewPos.z;

    return output;
}
