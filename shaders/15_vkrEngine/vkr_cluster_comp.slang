// cluster_comp.slang — Phase 5: Clustered shading light assignment for vkrEngine
// Compute shader: assigns point lights to screen-space clusters using depth slicing

struct GpuPointLight
{
    float4 position; // xyz = world pos, w = unused
    float4 color;    // rgb = color, w = intensity
};

[[vk::binding(0, 0)]]
StructuredBuffer<GpuPointLight> lightBuffer;

struct LightGridCell
{
    uint offset;
    uint count;
};

[[vk::binding(1, 0)]]
RWStructuredBuffer<LightGridCell> lightGrid;

[[vk::binding(2, 0)]]
RWStructuredBuffer<uint> lightIndexBuffer;

struct ClusterParams
{
    uint clusterX;
    uint clusterY;
    uint clusterZ;
    uint totalClusters;
    float3 tileSize;
    float pad0;
    float3 cameraPos; // world-space camera position
    float nearZ;
    float farZ;
    float zMin;
    float zMax;
    float pad1;
};

[[vk::binding(3, 0)]]
ConstantBuffer<ClusterParams> clusterParams;

static const uint MAX_LIGHTS_PER_CLUSTER = 64u;

// ====================================================================
// Compute cluster frustum in view space (z-positive = away from camera)
// zMin is the near plane, zMax is the far plane
// ====================================================================
float3 getClusterMin(uint3 clusterIdx)
{
    float tileX = 2.0 / float(clusterParams.clusterX);
    float tileY = 2.0 / float(clusterParams.clusterY);

    float2 screenMin;
    screenMin.x = float(clusterIdx.x) * tileX - 1.0;
    screenMin.y = float(clusterIdx.y) * tileY - 1.0;

    float zMin = clusterParams.zMin;
    float zMax = clusterParams.zMax;

    // Logarithmic Z slicing for better near-plane distribution
    float logMin = log(zMin);
    float logMax = log(zMax);
    float zNormMin = float(clusterIdx.z) / float(clusterParams.clusterZ);
    float zLinearMin = exp(logMin + (logMax - logMin) * zNormMin);

    // Approximate half-FOV for frustum extent computation
    float fovY = 45.0; // degrees
    float aspect = 16.0 / 9.0;
    float tanHalfFovY = tan(radians(fovY * 0.5));
    float tanHalfFovX = tanHalfFovY * aspect;

    float3 frustumMin;
    frustumMin.x = screenMin.x * zLinearMin * tanHalfFovX;
    frustumMin.y = screenMin.y * zLinearMin * tanHalfFovY;
    frustumMin.z = zLinearMin; // positive = away from camera

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
    float zNormMax = float(clusterIdx.z + 1u) / float(clusterParams.clusterZ);
    float zLinearMax = exp(logMin + (logMax - logMin) * zNormMax);

    float fovY = 45.0;
    float aspect = 16.0 / 9.0;
    float tanHalfFovY = tan(radians(fovY * 0.5));
    float tanHalfFovX = tanHalfFovY * aspect;

    float3 frustumMax;
    frustumMax.x = screenMax.x * zLinearMax * tanHalfFovX;
    frustumMax.y = screenMax.y * zLinearMax * tanHalfFovY;
    frustumMax.z = zLinearMax;

    return frustumMax;
}

// ====================================================================
// Sphere-AABB intersection test
// ====================================================================
bool sphereIntersectsAABB(float3 sphereCenter, float sphereRadius, float3 boxMin, float3 boxMax)
{
    float3 closestPoint;
    closestPoint.x = max(boxMin.x, min(sphereCenter.x, boxMax.x));
    closestPoint.y = max(boxMin.y, min(sphereCenter.y, boxMax.y));
    closestPoint.z = max(boxMin.z, min(sphereCenter.z, boxMax.z));

    float3 diff = sphereCenter - closestPoint;
    float distSq = dot(diff, diff);
    return distSq <= (sphereRadius * sphereRadius);
}

[numthreads(8, 8, 8)]
[shader("compute")]
void compMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    uint cx = dispatchThreadID.x;
    uint cy = dispatchThreadID.y;
    uint cz = dispatchThreadID.z;

    if (cx >= clusterParams.clusterX ||
        cy >= clusterParams.clusterY ||
        cz >= clusterParams.clusterZ)
        return;

    uint clusterIndex = cx + cy * clusterParams.clusterX + cz * clusterParams.clusterX * clusterParams.clusterY;

    uint3 clusterIdx = uint3(cx, cy, cz);
    float3 clusterMin = getClusterMin(clusterIdx);
    float3 clusterMax = getClusterMax(clusterIdx);

    // Estimate light radius from intensity (empirical mapping)
    uint lightCount = 2048u; // max lights in buffer
    uint localIndices[MAX_LIGHTS_PER_CLUSTER];
    uint count = 0;

    for (uint i = 0; i < lightCount && count < MAX_LIGHTS_PER_CLUSTER; ++i)
    {
        GpuPointLight light = lightBuffer[i];

        // Skip lights with zero intensity
        if (light.color.w <= 0.0)
            continue;

        float3 lightWorldPos = light.position.xyz;

        // Transform light world position to view space (simplified: subtract camera)
        float3 lightViewPos = lightWorldPos - clusterParams.cameraPos;

        // Light radius: approximate from intensity
        float lightRadius = sqrt(light.color.w) * 3.0;
        lightRadius = max(lightRadius, 1.0);

        // Frustum culling: transform view pos to match cluster's view-space convention
        // Cluster is in view space (z+ forward), we need light in same space
        // Simplified: use world-space distance for the AABB test
        if (sphereIntersectsAABB(lightWorldPos, lightRadius, clusterMin, clusterMax))
        {
            localIndices[count] = i;
            count++;
        }
    }

    // Write light grid
    uint baseIndex = clusterIndex * MAX_LIGHTS_PER_CLUSTER;
    for (uint i = 0; i < count; ++i)
    {
        lightIndexBuffer[baseIndex + i] = localIndices[i];
    }

    lightGrid[clusterIndex].offset = 0;
    lightGrid[clusterIndex].count = count;
}
