// deferred_gbuffer.slang — Phase 4: GBuffer pass for vkrEngine
// Renders Sponza geometry into MRT (Albedo, WorldNormal+Roughness, PBR params, Depth)

struct VSOutput
{
    float4 position : SV_Position;
    float3 worldPos : TEXCOORD0;
    float3 normal : TEXCOORD1;
    float2 texCoord : TEXCOORD2;
    float3 tangent : TEXCOORD3;
    float3 bitangent : TEXCOORD4;
};

// Set 0: Scene-level resources
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
ConstantBuffer<SceneUBO> sceneUbo;

struct CsmUBO
{
    float4x4 cascadeViewProj[4];
    float4 cascadeSplitDepths;
};
[[vk::binding(5, 0)]]
ConstantBuffer<CsmUBO> csmUbo;

// Set 1: Per-material resources
[[vk::binding(0, 1)]]
Texture2D baseColorTexture;
[[vk::binding(1, 1)]]
Texture2D normalTexture;
[[vk::binding(2, 1)]]
Texture2D metallicRoughnessTexture;
[[vk::binding(3, 1)]]
SamplerState materialSampler;

struct MaterialUBO
{
    float4 baseColorFactor;
    float metallicFactor;
    float roughnessFactor;
    float normalScale;
    float occlusionStrength;
    float3 emissiveFactor;
    float alphaCutoff;
    uint flags;
    uint pad0;
    uint pad1;
};
[[vk::binding(4, 1)]]
ConstantBuffer<MaterialUBO> materialUbo;

// Push constant: model matrix
struct PushBlock
{
    float4x4 modelMatrix;
};
[[vk::push_constant]]
PushBlock push;

// GBuffer MRT outputs
struct GBufferOutput
{
    float4 outAlbedo : SV_Target0;
    float4 outNormalRoughness : SV_Target1;
    float4 outPbr : SV_Target2;
};

[shader("vertex")]
VSOutput vertMain(
    float3 pos: POSITION,
    float3 normal: NORMAL,
    float2 texCoord: TEXCOORD0,
    float4 tangent: TANGENT)
{
    VSOutput output;

    float4 worldPos = mul(push.modelMatrix, float4(pos, 1.0));
    float4 viewPos = mul(sceneUbo.view, worldPos);
    output.position = mul(sceneUbo.projection, viewPos);
    output.worldPos = worldPos.xyz;

    float3x3 normalMatrix = (float3x3)push.modelMatrix;
    output.normal = normalize(mul(normalMatrix, normal));
    output.tangent = normalize(mul(normalMatrix, tangent.xyz));
    output.bitangent = cross(output.normal, output.tangent) * tangent.w;
    output.texCoord = texCoord;

    return output;
}

[shader("fragment")]
GBufferOutput fragMain(VSOutput input)
{
    bool hasBaseColorTex = (materialUbo.flags & 0x1u) != 0u;
    bool hasNormalTex = (materialUbo.flags & 0x2u) != 0u;
    bool hasMetallicRoughnessTex = (materialUbo.flags & 0x4u) != 0u;

    // --- Albedo ---
    float4 baseColorSample = hasBaseColorTex
                                 ? baseColorTexture.Sample(materialSampler, input.texCoord)
                                 : float4(1.0, 1.0, 1.0, 1.0);
    float3 albedo = baseColorSample.rgb * materialUbo.baseColorFactor.rgb;

    // --- Normal ---
    float3 N;
    if (hasNormalTex)
    {
        float3 tangentNormal = normalTexture.Sample(materialSampler, input.texCoord).rgb * 2.0 - 1.0;
        tangentNormal.xy *= materialUbo.normalScale;
        float3 T = normalize(input.tangent);
        float3 B = normalize(input.bitangent);
        float3 Ng = normalize(input.normal);
        N = normalize(tangentNormal.x * T + tangentNormal.y * B + tangentNormal.z * Ng);
    }
    else
    {
        N = normalize(input.normal);
    }

    // --- PBR params ---
    float2 mrSample = hasMetallicRoughnessTex
                          ? metallicRoughnessTexture.Sample(materialSampler, input.texCoord).bg
                          : float2(materialUbo.metallicFactor, materialUbo.roughnessFactor);
    float metallic = mrSample.x * materialUbo.metallicFactor;
    float roughness = max(mrSample.y * materialUbo.roughnessFactor, 0.04);
    float ao = materialUbo.occlusionStrength;

    GBufferOutput outp;
    outp.outAlbedo = float4(albedo, 1.0);
    // Encode world-space normal to [0,1], store roughness in alpha
    outp.outNormalRoughness = float4(N * 0.5 + 0.5, roughness);
    // r=metallic, g=ao
    outp.outPbr = float4(metallic, ao, 0.0, 0.0);

    return outp;
}
