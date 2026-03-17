struct SceneUBO
{
    float4x4 projection;
    float4x4 view;
    float3 camPos;
};
[[vk::binding(0, 0)]]
ConstantBuffer<SceneUBO> sceneUbo;

// Current frame depth (normalized device depth 0..1, Vulkan style)
[[vk::binding(1, 0)]]
Sampler2D depthTex;

// Current frame color (swapchain color after lighting pass)
[[vk::binding(2, 0)]]
Sampler2D colorTex;

struct SSRParams
{
    float maxRayDistance;
    float thickness;
    float stride;
    float intensity;

    float2 invResolution;
    int debugMode; // 0=off, 1=mask
    int maxSteps;
    float padding0;
};
[[vk::binding(3, 0)]]
ConstantBuffer<SSRParams> ssrParams;

struct VSOut
{
    float4 pos : SV_Position;
    float2 uv  : TEXCOORD0;
};

[shader("vertex")]
VSOut vertMain(uint vertexId : SV_VertexID)
{
    VSOut o;
    // Fullscreen triangle (flip Y for Vulkan)
    float2 pos;
    pos.x = (vertexId == 2) ? 3.0 : -1.0;
    pos.y = (vertexId == 1) ? -3.0 : 1.0;

    o.pos = float4(pos, 0.0, 1.0);
    o.uv = pos * 0.5 + 0.5;
    return o;
}

static float linearizeDepth(float z, float nearZ, float farZ)
{
    // Vulkan NDC depth is 0..1
    return (nearZ * farZ) / (farZ + z * (nearZ - farZ));
}

static float3 reconstructViewPos(float2 uv, float depth)
{
    // Reconstruct view position using inverse projection derived from projection matrix.
    // NOTE: This is an approximation: we solve clip->view using projection components.
    float4x4 P = sceneUbo.projection;
    float x = uv.x * 2.0 - 1.0;
    float y = uv.y * 2.0 - 1.0;

    // For a standard perspective matrix:
    // clip.x = view.x * P[0][0]
    // clip.y = view.y * P[1][1]
    // clip.z = view.z * P[2][2] + P[3][2]
    // clip.w = view.z
    // NDC = clip/clip.w
    float viewZ = P[3][2] / (depth - P[2][2]);
    float viewX = x * viewZ / P[0][0];
    float viewY = y * viewZ / P[1][1];
    return float3(viewX, viewY, viewZ);
}

static float3 estimateNormal(float2 uv)
{
    // Derive a normal from depth gradients in view space (cheap, noisy but ok for Level 7 baseline)
    float2 texel = ssrParams.invResolution;

    float dC = depthTex.Sample(uv).x;
    float dR = depthTex.Sample(uv + float2(texel.x, 0.0)).x;
    float dU = depthTex.Sample(uv + float2(0.0, texel.y)).x;

    float3 pC = reconstructViewPos(uv, dC);
    float3 pR = reconstructViewPos(uv + float2(texel.x, 0.0), dR);
    float3 pU = reconstructViewPos(uv + float2(0.0, texel.y), dU);

    float3 dx = pR - pC;
    float3 dy = pU - pC;
    return normalize(cross(dy, dx));
}

static float2 projectToUV(float3 viewPos)
{
    float4 clip = mul(sceneUbo.projection, float4(viewPos, 1.0));
    float2 ndc = clip.xy / max(clip.w, 1e-5);
    return ndc * 0.5 + 0.5;
}

static float raymarchSSR(float3 originVS, float3 dirVS, out float2 hitUV)
{
    float t = ssrParams.stride;
    float maxT = ssrParams.maxRayDistance;

    // Step along the reflection ray in view space, compare projected depth.
    for (int i = 0; i < ssrParams.maxSteps; ++i)
    {

        float3 p = originVS + dirVS * t;
        float2 uv = projectToUV(p);

        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0)
            return 0.0;

        float sceneDepth = depthTex.Sample(uv).x;
        float3 scenePos = reconstructViewPos(uv, sceneDepth);

        // In view space, z is negative forward typically. Compare along z.
        float dz = scenePos.z - p.z;
        if (abs(dz) < ssrParams.thickness)
        {
            hitUV = uv;
            return 1.0;
        }

        t += ssrParams.stride;
        if (t > maxT) break;
    }

    return 0.0;
}

[shader("fragment")]
float4 fragMain(VSOut i) : SV_TARGET
{
    float2 uv = i.uv;
    float depth = depthTex.Sample(uv).x;

    // background / sky
    if (depth >= 0.9999)
    {
        return float4(0.0, 0.0, 0.0, 0.0);
    }

    float3 posVS = reconstructViewPos(uv, depth);
    float3 N = estimateNormal(uv);

    // View direction (from point to camera) in view space points towards origin
    float3 V = normalize(-posVS);
    float3 R = normalize(reflect(-V, N));

    float2 hitUV = uv;
    float hit = raymarchSSR(posVS, R, hitUV);

    float3 refl = 0.0;
    if (hit > 0.5)
    {
        refl = colorTex.Sample(hitUV).xyz;
    }

    float edge = max(abs(uv.x - 0.5) * 2.0, abs(uv.y - 0.5) * 2.0);
    float fade = smoothstep(1.0, 0.85, edge);
    float3 outColor = refl * (ssrParams.intensity * fade);

    if (ssrParams.debugMode == 1)
    {
        return float4(hit.xxx, 1.0);
    }

    float3 baseColor = colorTex.Sample(uv).xyz;
    float3 finalColor = baseColor + outColor;
    return float4(finalColor, 1.0);
}
