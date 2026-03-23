struct TAAUParams
{
    float blendFactor;
    float reactiveClamp;
    float antiFlicker;
    float velocityScale;
    float historyClampGamma;
    float historyRejectThreshold;
    float pad0;
    float pad1;
};

[[vk::binding(0, 0)]]
Sampler2D inputColorTex;

[[vk::binding(1, 0)]]
Sampler2D historyColorTex;

[[vk::binding(2, 0)]]
Sampler2D velocityTex;

[[vk::binding(3, 0)]]
Sampler2D depthTex;

[[vk::binding(4, 0)]]
ConstantBuffer<TAAUParams> taauParams;

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

float3 sharpen(float3 c, float3 n0, float3 n1, float3 n2, float3 n3, float amount)
{
    float3 lap = (n0 + n1 + n2 + n3) * 0.25 - c;
    return c - lap * amount;
}

[shader("fragment")]
float4 fragMain(VSOut i) : SV_TARGET
{
    float2 uv = i.uv;

    float3 currentColor = inputColorTex.Sample(uv).xyz;
    float2 velocity = velocityTex.Sample(uv).xy;

    float2 historyUV = uv - velocity;
    historyUV = clamp(historyUV, float2(0.0), float2(1.0));
    float3 historyColor = historyColorTex.Sample(historyUV).xyz;

    uint w = 1;
    uint h = 1;
    inputColorTex.GetDimensions(w, h);
    float2 texel = 1.0 / float2(float(w), float(h));

    float3 n0 = inputColorTex.Sample(uv + float2( texel.x, 0.0)).xyz;
    float3 n1 = inputColorTex.Sample(uv + float2(-texel.x, 0.0)).xyz;
    float3 n2 = inputColorTex.Sample(uv + float2(0.0,  texel.y)).xyz;
    float3 n3 = inputColorTex.Sample(uv + float2(0.0, -texel.y)).xyz;
    float3 n4 = inputColorTex.Sample(uv + float2( texel.x,  texel.y)).xyz;
    float3 n5 = inputColorTex.Sample(uv + float2(-texel.x,  texel.y)).xyz;
    float3 n6 = inputColorTex.Sample(uv + float2( texel.x, -texel.y)).xyz;
    float3 n7 = inputColorTex.Sample(uv + float2(-texel.x, -texel.y)).xyz;

    float3 neighMin = min(currentColor, n0);
    neighMin = min(neighMin, n1);
    neighMin = min(neighMin, n2);
    neighMin = min(neighMin, n3);
    neighMin = min(neighMin, n4);
    neighMin = min(neighMin, n5);
    neighMin = min(neighMin, n6);
    neighMin = min(neighMin, n7);

    float3 neighMax = max(currentColor, n0);
    neighMax = max(neighMax, n1);
    neighMax = max(neighMax, n2);
    neighMax = max(neighMax, n3);
    neighMax = max(neighMax, n4);
    neighMax = max(neighMax, n5);
    neighMax = max(neighMax, n6);
    neighMax = max(neighMax, n7);

    float3 neighRange = max(neighMax - neighMin, float3(1e-4, 1e-4, 1e-4));
    float3 expandedMin = neighMin - neighRange * taauParams.historyClampGamma * 0.20;
    float3 expandedMax = neighMax + neighRange * taauParams.historyClampGamma * 0.20;
    float3 clampedHistory = clamp(historyColor, expandedMin, expandedMax);

    float lumaCurrent = dot(currentColor, float3(0.2126, 0.7152, 0.0722));
    float lumaHistory = dot(clampedHistory, float3(0.2126, 0.7152, 0.0722));
    float lumaDiff = abs(lumaCurrent - lumaHistory);

    float lumaN0 = dot(n0, float3(0.2126, 0.7152, 0.0722));
    float lumaN1 = dot(n1, float3(0.2126, 0.7152, 0.0722));
    float lumaN2 = dot(n2, float3(0.2126, 0.7152, 0.0722));
    float lumaN3 = dot(n3, float3(0.2126, 0.7152, 0.0722));

    float gradX = abs(lumaN0 - lumaN1);
    float gradY = abs(lumaN2 - lumaN3);
    float edgeStrength = saturate((gradX + gradY) * 6.0);

    float motionMag = length(velocity) * taauParams.velocityScale;
    float motionStability = saturate(1.0 - motionMag * 5.0);
    float colorStability = saturate(1.0 - lumaDiff * (5.0 * taauParams.reactiveClamp));
    float edgeStability = saturate(1.0 - edgeStrength * 0.9);

    // Per-pixel adaptive history weight:
    // - stable/static area -> strong accumulation
    // - motion / luma change / edge area -> reduce accumulation
    float stability = motionStability * colorStability * edgeStability;
    float minHistoryWeight = 0.16;
    float historyWeight = lerp(minHistoryWeight, taauParams.blendFactor, stability);

    // Color + velocity rejection only (no depth-based history rejection)
    float rejectThreshold = taauParams.historyRejectThreshold * lerp(0.75, 1.35, saturate(1.0 - stability));
    float colorReject = step(rejectThreshold, lumaDiff);
    historyWeight *= (1.0 - colorReject);

    // Static region boost, motion region suppress
    float staticBoost = saturate(stability * stability);
    historyWeight = lerp(historyWeight, min(historyWeight + 0.14, 0.96), staticBoost);
    historyWeight = clamp(historyWeight, 0.0, 0.96);

    float3 blended = lerp(currentColor, clampedHistory, historyWeight);
    float flickerDamp = saturate(taauParams.antiFlicker);
    float3 antiFlickerColor = lerp(currentColor, blended, flickerDamp);

    // Motion-adaptive sharpening:
    // - static + stable + non-edge region: allow moderate sharpening
    // - fast motion / unstable / strong edge: suppress sharpening to avoid reintroducing aliasing
    float motionSuppress = saturate(motionMag * 10.0);
    float edgeSuppress = saturate(edgeStrength * 0.9);
    float unstableSuppress = saturate(1.0 - stability);
    float sharpenGate = (1.0 - motionSuppress) * (1.0 - edgeSuppress) * (1.0 - unstableSuppress);
    sharpenGate = saturate(sharpenGate);

    float sharpenAmount = lerp(0.005, 0.065, sharpenGate);
    float3 sharpened = sharpen(antiFlickerColor, n0, n1, n2, n3, sharpenAmount);
    return float4(saturate(sharpened), 1.0);
}
