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
Sampler2D normalTex;

[[vk::binding(4, 0)]]
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

static float3 sampleNormalVS(float2 uv)
{
    // 从法线纹理读取世界法线（编码在 [0,1]），再转回视空间
    float3 nWorld = normalTex.Sample(uv).xyz * 2.0 - 1.0;
    nWorld = normalize(nWorld);

    float3x3 viewRot = (float3x3)sceneUbo.view;
    float3 nVS = normalize(mul(viewRot, nWorld));
    return nVS;
}

static float2 projectToUV(float3 viewPos)
{
    float4 clip = mul(sceneUbo.projection, float4(viewPos, 1.0));
    float2 ndc = clip.xy / max(clip.w, 1e-6);
    return ndc * 0.5 + 0.5;
}

static float hash12(float2 p)
{
    float h = dot(p, float2(127.1, 311.7));
    return frac(sin(h) * 43758.5453123);
}

static bool raymarchSSR(float3 originVS, float3 dirVS, float jitter01, out float2 hitUV, out float hitSteps)
{
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

        // 仅接受“ray 点在表面后方一点点”的命中，避免白块和穿帮
        if (dz > 0.0 && dz < ssrParams.thickness)
        {
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

    // Depth ：直接可视化主深度，使用线性化提升对比度
    if (ssrParams.debugMode == 3)
    {
        float nearPlane = max(sceneUbo.cameraPosNear.w, 1e-4);
        float farPlane = max(sceneUbo.cameraFarPadding.x, nearPlane + 1e-3);
        float zNdc = depth * 2.0 - 1.0;
        float linearDepth = (2.0 * nearPlane * farPlane) / max((farPlane + nearPlane - zNdc * (farPlane - nearPlane)), 1e-6);
        float linear01 = saturate((linearDepth - nearPlane) / (farPlane - nearPlane));

        // 近处亮、远处暗，便于看清立方体与地板层次
        float vis = 1.0 - linear01;
        vis = pow(vis, 0.65);
        return float4(vis, vis, vis, 1.0);
    }

    if (depth >= 0.9999)
    {
        return float4(0.0, 0.0, 0.0, 0.0);
    }

    float3 posVS = reconstructViewPos(uv, depth);
    float3 N = sampleNormalVS(uv);
    float3 V = normalize(-posVS);
    float3 R = normalize(reflect(-V, N));

    // 轻微偏移起点，减少地板自相交导致的遮挡/闪烁
    float3 rayOriginVS = posVS + N * max(ssrParams.thickness * 0.5, 0.01);

    float2 pixelCoord = uv / max(ssrParams.invResolution, float2(1e-6, 1e-6));
    float jitter01 = hash12(pixelCoord);

    float2 hitUV = uv;
    float hitSteps = 0.0;
    bool hit = raymarchSSR(rayOriginVS, R, jitter01, hitUV, hitSteps);

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

    // 命中稳定性过滤：
    // 1) 离屏幕四边至少留 2 像素，不在边缘采样，避免屏幕边缘采样污染
    // 2) 命中点若是天空/背景（depth 约等于 1）则视为无效
    float2 safeMin = ssrParams.invResolution * 2.0;
    float2 safeMax = 1.0 - safeMin;
    bool hitInBounds = hit && all(hitUV > safeMin) && all(hitUV < safeMax);
    float hitDepth = hitInBounds ? depthTex.Sample(hitUV).x : 1.0;
    bool hitValid = hitInBounds && (hitDepth < 0.9995);

    // 5-tap 邻域采样，降低条纹和闪烁
    float3 refl = 0.0;
    if (hitValid)
    {
        float2 t = ssrParams.invResolution;
        float3 s0 = colorTex.Sample(hitUV).xyz;
        float3 s1 = colorTex.Sample(hitUV + float2( t.x, 0.0)).xyz;
        float3 s2 = colorTex.Sample(hitUV + float2(-t.x, 0.0)).xyz;
        float3 s3 = colorTex.Sample(hitUV + float2(0.0,  t.y)).xyz;
        float3 s4 = colorTex.Sample(hitUV + float2(0.0, -t.y)).xyz;
        refl = (s0 * 0.40 + (s1 + s2 + s3 + s4) * 0.15);
    }

    float edge = max(abs(uv.x - 0.5) * 2.0, abs(uv.y - 0.5) * 2.0);
    float fade = smoothstep(1.0, 0.85, edge);

    // 仅让“白色地板(白色 cube 顶面)”拥有反射：
    // 1) 亮度高且低饱和（近白）
    // 2) 法线朝上（接近世界上方向）
    float luminance = dot(baseColor, float3(0.2126, 0.7152, 0.0722));
    float maxC = max(baseColor.r, max(baseColor.g, baseColor.b));
    float minC = min(baseColor.r, min(baseColor.g, baseColor.b));
    float saturation = maxC - minC;

    float whiteMask = smoothstep(0.55, 0.85, luminance) * (1.0 - smoothstep(0.10, 0.35, saturation));

    float3 worldUpVS = normalize(mul(sceneUbo.view, float4(0.0, 1.0, 0.0, 0.0)).xyz);
    float upDot = dot(normalize(N), worldUpVS);
    float floorMask = smoothstep(0.55, 0.90, upDot);

    // 使用 cameraPosNear/cameraFarPadding 里的 near/far 做距离衰减：
    // - 近处反射更明显
    // - 远处反射逐渐减弱，降低噪声和闪烁观感
    float nearPlane = max(sceneUbo.cameraPosNear.w, 1e-4);
    float farPlane = max(sceneUbo.cameraFarPadding.x, nearPlane + 1e-3);
    float viewDist = length(posVS);
    float dist01 = saturate((viewDist - nearPlane) / (farPlane - nearPlane));
    // 放宽远处衰减，避免两侧倒影“太浅/消失”
    float distanceFade = lerp(1.0, 0.7, dist01);

    float reflectMask = saturate(whiteMask * floorMask);

    // Fresnel-like 增强：掠射角更明显，正视角也保留较高基线，提升“可见性”
    float fresnel = pow(saturate(1.0 - dot(normalize(N), V)), 2.2);
    float reflectStrength = ssrParams.intensity * fade * reflectMask * distanceFade * (0.75 + 1.05 * fresnel);

    // 反射可见性增强：为了暗处也能看见
    float reflLuma = dot(refl, float3(0.2126, 0.7152, 0.0722));
    float darkSuppress = smoothstep(0.005, 0.06, reflLuma);
    reflectStrength *= max(0.35, darkSuppress);

    // 轻微提亮与对比增强，减少反射位置因光照不足看不见反射的问题
    float3 reflBoosted = pow(max(refl, 0.0), float3(0.9, 0.9, 0.9)) * 1.25;

    // 保底可见反射：即便反射源较暗，也给一点可感知贡献
    float minVisible = 0.04 * reflectMask * fade;

    float3 finalColor = baseColor + reflBoosted * reflectStrength + minVisible;
    return float4(finalColor, 1.0);
}
