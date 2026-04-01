struct SceneUBO
{
    float4x4 projection;
    float4x4 view;
    float3 camPos;
    float nearZ;
};

[[vk::binding(0, 0)]]
ConstantBuffer<SceneUBO> sceneUbo;

struct PointLight
{
    float4 position; // xyz: position, w: unused
    float4 color;    // rgb: color, w: intensity
};

[[vk::binding(1, 0)]]
StructuredBuffer<PointLight> lightBuffer;

struct LightGrid
{
    uint offset;
    uint count;
};

[[vk::binding(2, 0)]]
StructuredBuffer<LightGrid> lightGrid;

[[vk::binding(4, 0)]]
StructuredBuffer<uint> lightIndexBuffer;

struct ClusterParamsUBO
{
    uint clusterX;
    uint clusterY;
    uint clusterZ;
    uint totalClusters;
    float3 tileSize;
    float pad0;
    float3 cameraPos;
    float farZ;
    float zCount;
    float zMin;
    float zMax;
    float pad1;
    float pad2;
};

[[vk::binding(3, 0)]]
ConstantBuffer<ClusterParamsUBO> clusterParams;

struct VSInput
{
    float3 Pos : POSITION;
    float3 Normal : NORMAL;
    float2 texCoord : TEXCOORD;
};

struct VSOutput
{
    float4 Pos : SV_POSITION;
    float3 WorldPos : LOC1;
    float3 Normal : LOC2;
    float3 ViewPos : LOC4;
};

uint getClusterIndex(float3 viewPos)
{
    float2 screenUV = float2(
        (viewPos.x + 1.0) * 0.5,
        (viewPos.y + 1.0) * 0.5
    );

    uint tileX = clamp(uint(screenUV.x * clusterParams.clusterX), 0u, clusterParams.clusterX - 1u);
    uint tileY = clamp(uint(screenUV.y * clusterParams.clusterY), 0u, clusterParams.clusterY - 1u);

    float zMin = clusterParams.zMin;
    float zMax = clusterParams.zMax;
    float viewZ = -viewPos.z;

    float logMin = log(zMin);
    float logMax = log(zMax);
    float logZ = log(max(viewZ, zMin));

    float zNorm = clamp((logZ - logMin) / (logMax - logMin), 0.0, 1.0);
    uint tileZ = clamp(uint(zNorm * clusterParams.clusterZ), 0u, clusterParams.clusterZ - 1u);

    uint index = tileX + tileY * clusterParams.clusterX + tileZ * clusterParams.clusterX * clusterParams.clusterY;
    return index;
}

[[shader("vertex")]]
VSOutput vertMain(VSInput input, uint instanceID : SV_InstanceID)
{
    VSOutput output;

    PointLight light = lightBuffer[instanceID];

    float4 worldPos = float4(input.Pos * 0.3 + light.position.xyz, 1.0);
    output.WorldPos = worldPos.xyz;
    output.Normal = input.Normal;

    float4 viewPos = mul(sceneUbo.view, worldPos);
    float4 clipPos = mul(sceneUbo.projection, viewPos);
    output.Pos = clipPos;
    output.ViewPos = viewPos.xyz;

    return output;
}

[[shader("fragment")]]
float4 fragMain(VSOutput input) : SV_Target
{
    float3 worldPos = input.WorldPos;
    float3 N = normalize(input.Normal);
    float3 V = normalize(sceneUbo.camPos - worldPos);

    uint clusterIdx = getClusterIndex(input.ViewPos);
    LightGrid grid = lightGrid[clusterIdx];

    float3 color = float3(0.0, 0.0, 0.0);

    for (uint i = 0; i < grid.count; ++i)
    {
        uint lightIdx = lightIndexBuffer[grid.offset + i];
        PointLight light = lightBuffer[lightIdx];

        float3 L = light.position.xyz - worldPos;
        float dist = length(L);
        L = normalize(L);

        float3 lightColor = light.color.rgb * light.color.w;
        float attenuation = 1.0 / (1.0 + 0.09 * dist + 0.032 * dist * dist);
        float diff = max(dot(N, L), 0.0);
        float3 radiance = diff * lightColor * attenuation;

        color += radiance;
    }

    float ambient = 0.02;
    color += float3(ambient, ambient, ambient);

    color = color / (color + float3(1.0, 1.0, 1.0));
    color = pow(color, float3(0.4545, 0.4545, 0.4545));

    return float4(color, 1.0);
}
