struct SceneUBO
{
    float4x4 projection;
    float4x4 view;
    float3 camPos;
    float nearZ;
};
[[vk::binding(0, 0)]]
ConstantBuffer<SceneUBO> sceneUbo;

[[vk::binding(1, 0)]]
StructuredBuffer<float4x4> jointMatrices;

struct VSInput
{
    float3 Pos : POSITION;
    float3 Normal : NORMAL;
    uint4 jointIndices : BLENDINDICES0;
    float4 jointWeights : BLENDWEIGHT0;
};

struct VSOutput
{
    float4 Pos : SV_POSITION;
    float3 WorldPos : WORLD_POS;
    float3 Normal : NORMAL;
};

float3x3 rotationFromMat4(float4x4 m)
{
    return float3x3(
        m[0].xyz,
        m[1].xyz,
        m[2].xyz
    );
}

[[shader("vertex")]]
VSOutput vertMain(VSInput input)
{
    VSOutput output;

    float4x4 skinTransform = (float4x4)0;
    for (int i = 0; i < 4; i++)
    {
        float weight = input.jointWeights[i];
        if (weight > 0.0)
        {
            uint jointIdx = input.jointIndices[i];
            skinTransform += jointMatrices[jointIdx] * weight;
        }
    }

    float4 skinnedPos = mul(skinTransform, float4(input.Pos, 1.0));
    output.WorldPos = skinnedPos.xyz;

    float3x3 normalMat = rotationFromMat4(skinTransform);
    output.Normal = normalize(mul(normalMat, input.Normal));

    output.Pos = mul(sceneUbo.projection, mul(sceneUbo.view, skinnedPos));
    return output;
}

[[shader("fragment")]]
float4 fragMain(VSOutput input) : SV_Target0
{
    float3 N = normalize(input.Normal);
    float3 L = normalize(float3(1.0, 2.0, 1.5));
    float3 V = normalize(sceneUbo.camPos - input.WorldPos);

    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.0);

    float3 baseColor = float3(0.75, 0.65, 0.55);
    float metallic = 0.1;
    float roughness = 0.5;

    float3 F0 = lerp(float3(0.04), baseColor, metallic);

    float3 diffuse = baseColor * (1.0 - metallic) * NdotL;

    float3 H = normalize(L + V);
    float NdotH = max(dot(N, H), 0.0);
    float D = roughness * roughness;
    float denom = NdotH * NdotH * (D * D - 1.0) + 1.0;
    float specular = (D * D) / (3.14159 * denom * denom);

    float3 ambient = float3(0.03) * baseColor;

    float3 color = ambient + diffuse + F0 * specular * 0.25;

    color = color / (color + 1.0);
    color = pow(color, float3(1.0 / 2.2));

    return float4(color, 1.0);
}
