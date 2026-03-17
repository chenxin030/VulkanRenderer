struct SceneUBO
{
    float4x4 projection;
    float4x4 view;
    float4x4 invProjection;
    float4 cameraPosNear;
    float4 cameraFarPadding;
};
[[vk::binding(0, 0)]]
ConstantBuffer<SceneUBO> sceneUbo;

[[vk::binding(1, 0)]]
Sampler2D depthTex;

[[vk::binding(2, 0)]]
Sampler2D colorTex;

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
    float2 pos;
    pos.x = (vertexId == 2) ? 3.0 : -1.0;
    pos.y = (vertexId == 1) ? -3.0 : 1.0;

    o.pos = float4(pos, 0.0, 1.0);
    o.uv = pos * 0.5 + 0.5;
    return o;
}

static float3 reconstructViewPos(float2 uv, float depth)
{
    float4 clip = float4(uv * 2.0 - 1.0, depth, 1.0);
    float4 view = mul(sceneUbo.invProjection, clip);
    return view.xyz / max(view.w, 1e-6);
}

static float3 estimateNormal(float2 uv)
{
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
    float2 ndc = clip.xy / max(clip.w, 1e-6);
    return ndc * 0.5 + 0.5;
}

static bool raymarchSSR(float3 originVS, float3 dirVS, out float2 hitUV, out float hitSteps)
{
    float t = ssrParams.stride;
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

        if (abs(dz) < ssrParams.thickness)
        {
            float refineStart = lastT;
            float refineEnd = t;
            for (int r = 0; r < 4; ++r)
            {
                float mid = 0.5 * (refineStart + refineEnd);
                float3 mp = originVS + dirVS * mid;
                float2 muv = projectToUV(mp);
                float md = depthTex.Sample(muv).x;
                float3 mpos = reconstructViewPos(muv, md);
                float mdz = mpos.z - mp.z;
                if (mdz > 0.0)
                    refineStart = mid;
                else
                    refineEnd = mid;
            }
            float3 fp = originVS + dirVS * refineEnd;
            hitUV = projectToUV(fp);
            hitSteps = (float)i / max(1.0f, (float)ssrParams.maxSteps);
            return true;
        }

        lastT = t;
        t += ssrParams.stride;
        if (t > maxT)
            break;
    }

    return false;
}

[shader("fragment")]
float4 fragMain(VSOut i) : SV_TARGET
{
    float2 uv = i.uv;
    float depth = depthTex.Sample(uv).x;

    if (depth >= 0.9999)
    {
        return float4(0.0, 0.0, 0.0, 0.0);
    }

    float3 posVS = reconstructViewPos(uv, depth);
    float3 N = estimateNormal(uv);
    float3 V = normalize(-posVS);
    float3 R = normalize(reflect(-V, N));

    float2 hitUV = uv;
    float hitSteps = 0.0;
    bool hit = raymarchSSR(posVS, R, hitUV, hitSteps);

    if (ssrParams.debugMode == 1)
    {
        return float4(hit ? 1.0 : 0.0, 0.0, 0.0, 1.0);
    }
    if (ssrParams.debugMode == 2)
    {
        return float4(hitSteps, hitSteps, hitSteps, 1.0);
    }
    if (ssrParams.debugMode == 3)
    {
        return float4(depth.xxx, 1.0);
    }

    float3 baseColor = colorTex.Sample(uv).xyz;
    float3 refl = hit ? colorTex.Sample(hitUV).xyz : 0.0;

    float edge = max(abs(uv.x - 0.5) * 2.0, abs(uv.y - 0.5) * 2.0);
    float fade = smoothstep(1.0, 0.85, edge);

    float3 finalColor = baseColor + refl * (ssrParams.intensity * fade);
    return float4(finalColor, 1.0);
}
