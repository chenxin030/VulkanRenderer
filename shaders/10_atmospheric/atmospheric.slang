// Atmospheric scattering: Rayleigh + Mie scattering sky dome
// Based on "Accurate Atmospheric Scattering" by Sean O'Neil
// and "Precomputed Atmospheric Scattering" by Bruneton et al.
//
// Push constants (vertex + fragment):
//   invView, invProjection, cameraPos, viewportSize
//   sunElevation, sunAzimuth, timeOfDay, turbidity
//   mieCoefficient, rayleighCoefficient

struct PushConstants
{
    [[vk::offset(0)]]
    float4x4 invView;
    [[vk::offset(64)]]
    float4x4 invProjection;
    [[vk::offset(128)]]
    float4 cameraPos;
    [[vk::offset(144)]]
    float2 viewportSize;
    [[vk::offset(152)]]
    float2 _padding;          // matches C++ padding between vec2 and float
    [[vk::offset(160)]]
    float sunElevation;
    [[vk::offset(164)]]
    float sunAzimuth;
    [[vk::offset(168)]]
    float timeOfDay;
    [[vk::offset(172)]]
    float turbidity;
    [[vk::offset(176)]]
    float mieCoefficient;
    [[vk::offset(180)]]
    float rayleighCoefficient;
};

[[vk::push_constant]]
PushConstants pc;

static const float3 SUN_COLOR = float3(1.0f, 0.97f, 0.89f);
static const float3 EARTH_POS = float3(0.0f, -6356.0f, 0.0f);
static const float ATMOSPHERE_HEIGHT = 6420.0f - 6356.0f; // 64km
static const float RAYLEIGH_SCALE = 8.0f;
static const float MIE_SCALE = 1.2f;

float3 sphereIntersection(float3 rayStart, float3 rayDir, float3 spherePos, float sphereRadius)
{
    float3 oc = rayStart - spherePos;
    float b = dot(oc, rayDir);
    float c = dot(oc, oc) - sphereRadius * sphereRadius;
    float disc = b * b - c;
    if (disc < 0.0f) return float3(-1.0f);
    float sqrtDisc = sqrt(disc);
    float t0 = -b - sqrtDisc;
    float t1 = -b + sqrtDisc;
    return float3(t0, t1, disc >= 0.0f ? 1.0f : 0.0f);
}

float3 computeIncidentLight(float3 rayStart, float3 rayDir, float3 sunDir, float turbidity)
{
    // Atmospheric parameters derived from turbidity and coefficients
    float RayleighScale = RAYLEIGH_SCALE;
    float MieScale = MIE_SCALE;
    float RayleighCoefficient = 0.000058f;
    float MieCoefficient = 0.000021f;

    float RayleighPhase = 0.059683f * (1.0f + pow(dot(rayDir, sunDir), 2.0f));
    float MiePhase = 0.119366f * pow(1.0f - pow(0.78f, 2.0f), -1.5f) / (1.0f + 1.5f * dot(rayDir, sunDir));

    float cameraHeight = rayStart.y - EARTH_POS.y;

    float innerRadius = length(EARTH_POS);
    float outerRadius = innerRadius + ATMOSPHERE_HEIGHT;

    float3 cameraPos = rayStart - EARTH_POS;
    float3 hit = sphereIntersection(cameraPos, rayDir, float3(0, 0, 0), outerRadius);

    float far = hit.y;

    float cameraScale = 1.0f / (ATMOSPHERE_HEIGHT);

    float samples = 2.0f;
    float sampleStep = far / samples;
    float scaledStep = (far / samples) / RAYLEIGH_SCALE;

    float RayleighDepth = 0.0f;
    float MieDepth = 0.0f;

    float scaledLength = sampleStep * RAYLEIGH_SCALE;

    for (float i = 0.0f; i < samples; i += 1.0f)
    {
        float temp = length(cameraPos + rayDir * (i + 0.5f) * sampleStep) - innerRadius;
        float height = temp * cameraScale;
        float hr = exp(-height * 0.5f) * scaledLength;
        float hm = exp(-height * 0.15f) * scaledLength;
        RayleighDepth += hr;
        MieDepth += hm;
    }

    float3 extinction = exp(-(RayleighCoefficient * RayleighDepth + MieCoefficient * MieDepth) * 1000.0f);

    float farCamera = far;
    float cameraRayleighDepth = 0.0f;
    float cameraRayleighStep = farCamera / 16.0f;
    float scaledCameraStep = cameraRayleighStep * RAYLEIGH_SCALE;

    for (float i = 0.0f; i < 16.0f; i += 1.0f)
    {
        float temp = length(cameraPos + rayDir * (i + 0.5f) * cameraRayleighStep) - innerRadius;
        float height = max(temp * cameraScale, 0.0f);
        cameraRayleighDepth += exp(-height * 0.5f) * scaledCameraStep;
    }

    float sunRayLength = far / 8.0f;
    float sunRayleighDepth = 0.0f;
    float scaledSunStep = sunRayLength * RAYLEIGH_SCALE;

    for (float i = 0.0f; i < 8.0f; i += 1.0f)
    {
        float temp = length(cameraPos + sunDir * (i + 0.5f) * sunRayLength) - innerRadius;
        float height = max(temp * cameraScale, 0.0f);
        sunRayleighDepth += exp(-height * 0.5f) * scaledSunStep;
    }

    float sunEx = exp(-(cameraRayleighDepth + sunRayleighDepth) * RayleighCoefficient * 1000.0f);

    return extinction * SUN_COLOR * (RayleighPhase * sunEx + MiePhase * sunEx);
}

struct VSOutput
{
    float4 Pos : SV_POSITION;
    float2 uv;
    float3 worldDir;
};

[[shader("vertex")]]
VSOutput vertMain(uint vertexID : SV_VertexID)
{
    VSOutput output;

    float2 pos;
    if (vertexID == 0)
    {
        pos = float2(-1.0, -1.0);
    }
    else if (vertexID == 1)
    {
        pos = float2(3.0, -1.0);
    }
    else
    {
        pos = float2(-1.0, 3.0);
    }

    output.Pos = float4(pos, 1.0, 1.0);
    output.uv = pos * 0.5 + 0.5;

    // Reconstruct world-space ray direction from screen position
    float4 clipPos = float4(pos, 0.5, 1.0);
    float4 worldPos4 = mul(pc.invView, mul(pc.invProjection, clipPos));
    output.worldDir = normalize(worldPos4.xyz / worldPos4.w - pc.cameraPos.xyz);

    return output;
}

[[shader("fragment")]]
float4 fragMain(VSOutput input) : SV_Target0
{
    float3 rayDir = normalize(input.worldDir);

    // Sun direction from elevation (angle above horizon) and azimuth
    float3 sunDir = normalize(float3(
        cos(pc.sunElevation) * sin(pc.sunAzimuth),
        sin(pc.sunElevation),
        cos(pc.sunElevation) * cos(pc.sunAzimuth)
    ));

    float3 cameraPos = pc.cameraPos.xyz;

    // Compute atmospheric scattering
    float3 skyColor = computeIncidentLight(cameraPos, rayDir, sunDir, pc.turbidity);

    // Sky gradient: blend to horizon color based on ray direction
    float horizonBlend = smoothstep(-0.05f, 0.2f, rayDir.y);
    float3 horizonColor = float3(0.6f, 0.75f, 0.85f) * 0.5f;

    skyColor = lerp(horizonColor, skyColor, horizonBlend);

    // Sun disc
    float sunDot = dot(rayDir, sunDir);
    float sunDisc = smoothstep(0.9995f, 0.9999f, sunDot);
    float sunGlow = pow(max(sunDot, 0.0f), 256.0f);
    skyColor += SUN_COLOR * sunDisc * 5.0f;
    skyColor += float3(1.0f, 0.8f, 0.5f) * sunGlow * 0.5f;

    // Ground below horizon
    if (rayDir.y < 0.0f)
    {
        float groundT = -rayDir.y * 10.0f;
        float3 groundColor = float3(0.15f, 0.12f, 0.1f) * 0.5f;
        skyColor = lerp(skyColor, groundColor, min(groundT, 1.0f));
    }

    // HDR tone mapping
    skyColor = skyColor / (skyColor + 1.0f);

    return float4(skyColor, 1.0f);
}
