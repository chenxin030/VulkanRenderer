struct SceneUBO
{
    float4x4 projection;
    float4x4 view;
    float3 camPos;
    float nearZ;
};

[[vk::binding(3, 0)]]
ConstantBuffer<SceneUBO> sceneUbo;

struct PointLight
{
    float4 position;
    float4 color;
};

[[vk::binding(0, 0)]]
StructuredBuffer<PointLight> lightBuffer;

struct LightGrid
{
    uint offset;
    uint count;
};

[[vk::binding(1, 0)]]
RWStructuredBuffer<LightGrid> lightGrid;

[[vk::binding(2, 0)]]
RWStructuredBuffer<uint> lightIndexBuffer;

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

[[vk::binding(4, 0)]]
ConstantBuffer<ClusterParamsUBO> clusterParams;

static const uint MAX_LIGHTS_PER_CLUSTER = 64u;

float3 getClusterMin(uint3 clusterIdx)
{
    float tileX = 2.0 / float(clusterParams.clusterX);
    float tileY = 2.0 / float(clusterParams.clusterY);

    float2 screenMin;
    screenMin.x = float(clusterIdx.x) * tileX - 1.0;
    screenMin.y = float(clusterIdx.y) * tileY - 1.0;

    float zMin = clusterParams.zMin;
    float zMax = clusterParams.zMax;
    float logMin = log(zMin);
    float logMax = log(zMax);
    float clusterZNorm = float(clusterIdx.z) / float(clusterParams.clusterZ);
    float zNormMin = clusterZNorm;
    float zNormMax = (float(clusterIdx.z) + 1.0) / float(clusterParams.clusterZ);

    float zLinearMin = exp(lerp(logMin, logMax, zNormMin));
    float zLinearMax = exp(lerp(logMin, logMax, zNormMax));

    zLinearMin = max(zLinearMin, zMin);
    zLinearMax = min(zLinearMax, zMax);

    float aspect = 16.0 / 9.0;
    float fovY = 45.0;
    float tanHalfFovY = tan(radians(fovY * 0.5));
    float tanHalfFovX = tanHalfFovY * aspect;

    float3 frustumMin;
    frustumMin.x = screenMin.x * zLinearMin * tanHalfFovX;
    frustumMin.y = screenMin.y * zLinearMin * tanHalfFovY;
    frustumMin.z = -zLinearMin;

    return frustumMin;
}

float3 getClusterMax(uint3 clusterIdx)
{
    float tileX = 2.0 / float(clusterParams.clusterX);
    float tileY = 2.0 / float(clusterParams.clusterY);

    float2 screenMax;
    screenMax.x = float(clusterIdx.x + 1u) * tileX - 1.0;
    screenMax.y = float(clusterIdx.y + 1u) * tileY - 1.0;

    float zMin = clusterParams.zMin;
    float zMax = clusterParams.zMax;
    float logMin = log(zMin);
    float logMax = log(zMax);
    float clusterZNorm = (float(clusterIdx.z) + 1.0) / float(clusterParams.clusterZ);

    float zLinearMax = exp(lerp(logMin, logMax, clusterZNorm));
    zLinearMax = min(zLinearMax, zMax);

    float aspect = 16.0 / 9.0;
    float fovY = 45.0;
    float tanHalfFovY = tan(radians(fovY * 0.5));
    float tanHalfFovX = tanHalfFovY * aspect;

    float3 frustumMax;
    frustumMax.x = screenMax.x * zLinearMax * tanHalfFovX;
    frustumMax.y = screenMax.y * zLinearMax * tanHalfFovY;
    frustumMax.z = -zLinearMax;

    return frustumMax;
}

bool sphereIntersectsAABB(float3 sphereCenter, float sphereRadius, float3 boxMin, float3 boxMax)
{
    float3 closestPoint;
    closestPoint.x = max(boxMin.x, min(sphereCenter.x, boxMax.x));
    closestPoint.y = max(boxMin.y, min(sphereCenter.y, boxMax.y));
    closestPoint.z = max(boxMin.z, min(sphereCenter.z, boxMax.z));

    float3 diff = sphereCenter - closestPoint;
    float distSquared = dot(diff, diff);

    return distSquared <= (sphereRadius * sphereRadius);
}

[numthreads(8, 8, 8)]
[shader("compute")]
void compMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint clusterX = dispatchThreadID.x;
    uint clusterY = dispatchThreadID.y;
    uint clusterZ = dispatchThreadID.z;

    if (clusterX >= clusterParams.clusterX ||
        clusterY >= clusterParams.clusterY ||
        clusterZ >= clusterParams.clusterZ)
    {
        return;
    }

    uint3 clusterIdx = uint3(clusterX, clusterY, clusterZ);
    uint clusterIndex = clusterIdx.x +
                        clusterIdx.y * clusterParams.clusterX +
                        clusterIdx.z * clusterParams.clusterX * clusterParams.clusterY;

    float3 clusterMin = getClusterMin(clusterIdx);
    float3 clusterMax = getClusterMax(clusterIdx);

    uint lightCount = clusterParams.totalClusters > 0 ? 2048 : 0;
    uint offset = 0;
    uint count = 0;

    uint localIndices[MAX_LIGHTS_PER_CLUSTER];

    for (uint i = 0; i < lightCount && count < MAX_LIGHTS_PER_CLUSTER; ++i)
    {
        PointLight light = lightBuffer[i];
        float3 lightPos = light.position.xyz;

        float lightRadius = sqrt(light.color.w) * 0.5;
        if (lightRadius < 0.1)
        {
            lightRadius = 2.0;
        }

        if (sphereIntersectsAABB(lightPos, lightRadius, clusterMin, clusterMax))
        {
            localIndices[count] = i;
            count++;
        }
    }

    lightGrid[clusterIndex].offset = offset;
    lightGrid[clusterIndex].count = count;

    uint baseIndex = clusterIndex * MAX_LIGHTS_PER_CLUSTER;
    for (uint i = 0; i < count; ++i)
    {
        lightIndexBuffer[baseIndex + i] = localIndices[i];
    }
}
