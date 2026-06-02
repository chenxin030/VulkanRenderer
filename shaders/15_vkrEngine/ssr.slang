// ssr.slang — Phase 4: SSR compute pass for vkrEngine
// Full-screen triangle, reads GBuffer normal+depth+color, outputs SSR reflection

struct SceneUBO
{
    float4x4 projection;
    float4x4 view;
    float4x4 invProjection;
    float4 cameraPosNear;    // xyz = camPos, w = near
    float4 cameraFarPadding; // x = far
};
[[vk::binding(0, 0)]]
ConstantBuffer<SceneUBO> sceneUbo;

[[vk::binding(1, 0)]]
Sampler2D depthTex;

[[vk::binding(2, 0)]]
Sampler2D colorTex;

[[vk::binding(3, 0)]]
Sampler2D normalTex;

struct SSRParams
{
    float maxRayDistance;
    float thickness;
    float stride;
    float intensity;
    float2 invResolution;
    int debugMode;
    int maxSteps;
    float padding0;
};
[[vk::binding(4, 0)]]
ConstantBuffer<SSRParams> ssrParams;

struct VSOut
{
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
};

float3 reconstructViewPos(float2 uv, float depth)
{
    float4 clip = float4(uv * 2.0 - 1.0, depth, 1.0);
    float4 view = mul(sceneUbo.invProjection, clip);
    return view.xyz / max(view.w, 1e-6);
}

float3 sampleNormalVS(float2 uv, out float outRoughness)
{
    float4 normalData = normalTex.Sample(uv);
    outRoughness = normalData.a;
    float3 nWorld = normalData.xyz * 2.0 - 1.0;
    nWorld = normalize(nWorld);
    float3x3 viewRot = (float3x3)sceneUbo.view;
    float3 nVS = normalize(mul(viewRot, nWorld));
    return nVS;
}

float2 projectToUV(float3 viewPos)
{
    float4 clip = mul(sceneUbo.projection, float4(viewPos, 1.0));
    float2 ndc = clip.xy / max(clip.w, 1e-6);
    return ndc * 0.5 + 0.5;
}

float hash12(float2 p)
{
    float h = dot(p, float2(127.1, 311.7));
    return frac(sin(h) * 43758.5453123);
}

bool raymarchSSR(float3 originVS, float3 dirVS, float jitter01, out float2 hitUV, out float hitSteps)
{
    hitUV = float2(0.0, 0.0);
    hitSteps = 0.0;
    float t = ssrParams.stride * (0.5 + jitter01);
    float maxT = ssrParams.maxRayDistance;
    float lastT = 0.0;

    for (int i = 0; i < ssrParams.maxSteps; ++i)
    {
        float3 p = originVS + dirVS * t;
        float2 uv = projectToUV(p);

        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0)
            return false;

        float sceneDepth = depthTex.Sample(uv).x;
        float3 scenePos = reconstructViewPos(uv, sceneDepth);
        float dz = scenePos.z - p.z;

        if (dz > 0.0 && dz < ssrParams.thickness)
        {
            // Refine with binary search
            float refineStart = lastT;
            float refineEnd = t;
            for (int r = 0; r < 5; ++r)
            {
                float mid = 0.5 * (refineStart + refineEnd);
                float3 mp = originVS + dirVS * mid;
                float2 muv = projectToUV(mp);
                float md = depthTex.Sample(muv).x;
                float3 mpos = reconstructViewPos(muv, md);
                float mdz = mpos.z - mp.z;
                if (mdz > 0.0)
                    refineEnd = mid;
                else
                    refineStart = mid;
            }
            float3 fp = originVS + dirVS * refineEnd;
            hitUV = projectToUV(fp);
            hitSteps = (float)i / max(1.0, (float)ssrParams.maxSteps);
            return true;
        }

        lastT = t;
        t += ssrParams.stride;
        if (t > maxT)
            break;
    }

    return false;
}

[shader("vertex")]
VSOut vertMain(uint vertexId: SV_VertexID)
{
    VSOut o;
    float2 pos;
    pos.x = (vertexId == 2u) ? 3.0 : -1.0;
    pos.y = (vertexId == 1u) ? -3.0 : 1.0;
    o.pos = float4(pos, 0.0, 1.0);
    o.uv = pos * 0.5 + 0.5;
    return o;
}

[shader("fragment")]
float4 fragMain(VSOut i) : SV_TARGET
{
    float2 uv = i.uv;
    float depth = depthTex.Sample(uv).x;

    // Debug: depth visualization
    if (ssrParams.debugMode == 3)
    {
        float3 vp = reconstructViewPos(uv, depth);
        float linearDepth = -vp.z / sceneUbo.cameraFarPadding.x;
        return float4(linearDepth.xxx, 1.0);
    }

    // Debug: normal visualization
    if (ssrParams.debugMode == 4)
    {
        float rgh;
        float3 nVS = sampleNormalVS(uv, rgh);
        return float4(nVS * 0.5 + 0.5, 1.0);
    }

    // Skip sky
    if (depth >= 1.0)
        return float4(0.0, 0.0, 0.0, 1.0);

    float roughness;
    float3 nVS = sampleNormalVS(uv, roughness);

    // Skip very rough surfaces (no visible reflections)
    if (roughness > ssrParams.thickness * 8.0)
        return float4(0.0, 0.0, 0.0, 1.0);

    float3 viewPos = reconstructViewPos(uv, depth);
    float3 V = normalize(-viewPos);

    // Reflect view direction about surface normal
    float3 R = reflect(-V, nVS);

    float jitter = hash12(uv * 100.0 + float2(0.1, 0.2));

    float2 hitUV;
    float hitSteps;
    bool hit = raymarchSSR(viewPos, R, jitter, hitUV, hitSteps);

    float3 reflectionColor = float3(0.0, 0.0, 0.0);

    if (hit)
    {
        reflectionColor = colorTex.Sample(hitUV).rgb;

        // Fade at screen edges
        float2 edgeFade = 1.0 - abs(hitUV * 2.0 - 1.0);
        float edgeFactor = smoothstep(0.0, 0.15, edgeFade.x) * smoothstep(0.0, 0.15, edgeFade.y);

        // Fade by roughness
        float roughnessFade = 1.0 - roughness;

        // Fade at max distance
        float distFade = 1.0 - (hitSteps);

        float fade = edgeFactor * roughnessFade * distFade * ssrParams.intensity;

        // Debug: hit visualization
        if (ssrParams.debugMode == 1)
            return float4(0.0, 1.0, 0.0, 1.0);
        if (ssrParams.debugMode == 2)
            return float4(hitSteps.xxx, 1.0);

        return float4(reflectionColor * fade, 1.0);
    }

    // Debug: miss visualization
    if (ssrParams.debugMode == 1)
        return float4(1.0, 0.0, 0.0, 1.0);

    return float4(0.0, 0.0, 0.0, 1.0);
}
