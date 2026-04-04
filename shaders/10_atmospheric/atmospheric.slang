// Atmospheric scattering: sky dome shader
// Based on Bruneton "A Scalable and Production Ready Sky and Atmosphere Rendering Technique"
// and O'Neil "Accurate Atmospheric Scattering" (GPU Gems 2, Ch. 16)
//
// Key improvements over basic O'Neil single-scatter:
//   1. Transmittance LUT (precomputed by compute shader) — eliminates secondary ray marching
//   2. Ozone absorption — blue sky near horizon at sunset
//   3. Correct Henyey-Greenstein Mie phase function
//   4. Multi-scattering approximation (isotropic)
//   5. Physical sun disc with extinction
//
// Push constant layout (total 224 bytes = 14 * 16):
//   [0]   mat4 invView        (64)
//   [64]  mat4 invProjection (64)
//   [128] vec4 cameraPos     (16)
//   [144] vec2 viewportSize  (8)
//   [152] vec2 sunDir        (8)   — normalized, world space
//   [160] vec4 atmParams     (16) — innerRadius, outerRadius, RayleighCoeff, MieCoeff
//   [176] vec4 scaleParams   (16) — RayleighScale, MieScale, g, ozoneCoeff
//   [192] vec4 multiParams   (16) — Rayleigh4PI, Mie4PI, RayleighESun, MieESun
//   [208] vec2 ozoneHeight   (8)
//   [216] vec2 pad           (8)   — padding to 224 bytes total

struct PushConstants
{
    [[vk::offset(0)]]
    float4x4 invView;
    [[vk::offset(64)]]
    float4x4 invProjection;
    [[vk::offset(128)]]
    float4 cameraPos;      // xyz = camera world position (km), w = unused
    [[vk::offset(144)]]
    float2 viewportSize;
    [[vk::offset(152)]]
    float2 sunDir;         // x = elevation (radians), y = azimuth (radians)
    [[vk::offset(160)]]
    float4 atmParams;      // x=innerRadius(km), y=outerRadius(km), z=RayleighCoeff(km⁻¹), w=MieCoeff(km⁻¹)
    [[vk::offset(176)]]
    float4 scaleParams;    // x=RayleighScale(km), y=MieScale(km), z=g(Mie anisotropy), w=ozoneCoeff(km⁻¹)
    [[vk::offset(192)]]
    float4 esunParams;     // x=Rayleigh4PI, y=Mie4PI, z=RayleighESun, w=MieESun
    [[vk::offset(208)]]
    float2 ozoneHeight;   // x=ozone peak altitude(km), y=unused
    [[vk::offset(216)]]
    float2 pad;
};

[[vk::push_constant]]
PushConstants pc;

#define INNER_RADIUS  (pc.atmParams.x)
#define OUTER_RADIUS  (pc.atmParams.y)
#define ATM_HEIGHT    (OUTER_RADIUS - INNER_RADIUS)
#define BETA_R        (pc.atmParams.z)
#define BETA_M        (pc.atmParams.w)
#define H_R           (pc.scaleParams.x)
#define H_M           (pc.scaleParams.y)
#define MIE_G         (pc.scaleParams.z)
#define BETA_O3       (pc.scaleParams.w)
#define OZONE_H       (pc.ozoneHeight.x)

static const float3 SUN_SPECTRAL_COLOR = float3(1.0f, 0.97f, 0.89f);
static const float SUN_ANGULAR_RADIUS   = 0.00925f; // 0.53 degrees
static const float SUN_ANGULAR_COS      = 0.999957f; // cos(0.53°)

// ----------------------------------------------------------
// Density / extinction helpers
// ----------------------------------------------------------

float rayleighDensity(float h)
{
    return exp(-h / H_R);
}

float mieDensity(float h)
{
    return exp(-h / H_M);
}

// Ozone: triangular profile peaking at OZONE_H km
// sigma_O3(h) = BETA_O3 * max(0, 1 - |h - 25| / 15)
float ozoneDensity(float h)
{
    if (BETA_O3 <= 0.0f) return 0.0f;
    return BETA_O3 * max(0.0f, 1.0f - abs(h - OZONE_H) / 15.0f);
}

// Rayleigh + Mie scattering at height h
float2 scattering(float h)
{
    return float2(
        BETA_R * rayleighDensity(h),
        BETA_M * mieDensity(h)
    );
}

// Total extinction (Rayleigh + Mie + ozone)
float3 extinction(float h)
{
    return float3(
        BETA_R * rayleighDensity(h) + BETA_M * mieDensity(h) + ozoneDensity(h),
        BETA_M * mieDensity(h),
        ozoneDensity(h)
    );
}

// ----------------------------------------------------------
// Phase functions
// ----------------------------------------------------------

// Rayleigh phase: 3/(16π) * (1 + cos²θ)
float rayleighPhase(float cosTheta)
{
    return 0.059683f * (1.0f + cosTheta * cosTheta);
}

// Mie phase: Henyey-Greenstein P(θ) = (1-g²) / (4π·(1+g²-2g·cosθ)^1.5)
float miePhase(float cosTheta)
{
    float g2 = MIE_G * MIE_G;
    float denom = pow(1.0f + g2 - 2.0f * MIE_G * cosTheta, 1.5f);
    return 0.119366f * (1.0f - g2) / denom;
}

// Isotropic phase for multi-scattering
static const float ISO_PHASE = 0.079577f; // 1/(4π)

// ----------------------------------------------------------
// Ray-sphere intersection
// Returns (t_near, t_far). t_near<0 means ray origin is inside.
// ----------------------------------------------------------
float2 sphereHit(float3 ro, float3 rd, float radius)
{
    float3 oc = ro;
    float b = dot(oc, rd);
    float c = dot(oc, oc) - radius * radius;
    float disc = b * b - c;
    if (disc < 0.0f) return float2(-1.0f);
    float sd = sqrt(disc);
    return float2(-b - sd, -b + sd);
}

// Distance from ro along rd that exits the atmosphere (positive t)
float atmosphereExitDist(float3 ro, float3 rd)
{
    float2 outerHit = sphereHit(ro, rd, OUTER_RADIUS);
    float tNear = outerHit.x;
    float tFar  = outerHit.y;
    if (tFar < 0.0f) return 0.0f;       // pointing away
    if (tNear < 0.0f) return max(tFar, 0.0f); // inside atmosphere
    return max(tFar - tNear, 0.0f);
}

// ----------------------------------------------------------
// Transmittance LUT helpers
// ----------------------------------------------------------
// LUT dimensions (must match C++ constants)
static const float LUT_W = 256.0f;
static const float LUT_H =   64.0f;

float2 lutUVFromHeightMu(float hKm, float mu)
{
    float u_r = clamp(hKm / ATM_HEIGHT, 0.0f, 1.0f);
    float u_mu = clamp(mu * 0.5f + 0.5f, 0.0f, 1.0f);
    return float2(u_mu, u_r);
}

// T(r, mu) from LUT: height in km, mu = cos(view_angle_from_vertical)
float3 T_lut(float hKm, float mu)
{
    float u_mu = clamp(mu * 0.5f + 0.5f, 0.001f, 0.999f);
    float u_r  = clamp(hKm / ATM_HEIGHT, 0.001f, 0.999f);
    return sampleLUT(float2(u_mu, u_r));
}

// ----------------------------------------------------------
// Main scattering function
// ----------------------------------------------------------
float3 computeSkyRadiance(float3 rayStart, float3 rayDir, float3 sunDir)
{
    const int PRIMARY_STEPS = 16;

    // Distance to exit atmosphere along view ray
    float rayLen = atmosphereExitDist(rayStart, rayDir);
    if (rayLen <= 0.0f) return float3(0.0f);

    float stepSize = rayLen / float(PRIMARY_STEPS);
    float cosTheta = dot(rayDir, sunDir);

    // Phase functions (constant per-pixel for given sun direction)
    float phaseR = rayleighPhase(cosTheta);
    float phaseM = miePhase(cosTheta);

        // Camera height in km — pc.cameraPos.xyz is already in planet-centered km coords
    float hCam = length(pc.cameraPos.xyz) - INNER_RADIUS;
    hCam = max(hCam, 0.0f);

    // Camera-to-sun mu (for T(camera → sun) lookup)
    float muCamSun = dot(normalize(pc.cameraPos.xyz), sunDir);

    // Precompute sun-side transmittance at camera height (constant per pixel)
    float3 T_cam_to_sun = T_lut(hCam, muCamSun);

    float3 scatterR = float3(0.0f);
    float3 scatterM = float3(0.0f);

    for (int i = 0; i < PRIMARY_STEPS; i++)
    {
        float t = (float(i) + 0.5f) * stepSize;
        float3 p = rayStart + rayDir * t;

        float r = length(p);
        float h = r - INNER_RADIUS;
        h = max(h, 0.0f);

        // Scattering coefficients at this height
        float2 scat = scattering(h);

        // mu of sun ray at sample point
        float muSun = dot(normalize(p), sunDir);

        // Transmittance from sample to sun
        float3 T_sample_to_sun = T_lut(h, muSun);

        // Transmittance from camera to sample point (along view ray)
        float3 T_cam_to_sample = T_lut(hCam, dot(normalize(rayStart), rayDir));

        float3 combinedT = T_cam_to_sample * T_sample_to_sun;

        scatterR += scat.x * phaseR * combinedT * stepSize;
        scatterM += scat.y * phaseM * combinedT * stepSize;
    }

    // Multi-scattering approximation (Bruneton)
    // L_ms = L_ss / (1 - F_ms) with F_ms ≈ 0.1
    float F_ms = 0.10f;
    float3 L_ss = scatterR + scatterM;
    float3 L_ms = L_ss * F_ms / (1.0f - F_ms);
    float3 inScatter = L_ss + L_ms;

    // Sun color attenuated along the full view ray
    float3 sunColor = SUN_SPECTRAL_COLOR * T_cam_to_sun;

    return inScatter * sunColor;
}

// ----------------------------------------------------------
// Sun disc rendering
// ----------------------------------------------------------
float3 renderSun(float3 rayDir, float3 sunDir)
{
    float cosAngle = dot(rayDir, sunDir);

    // Physical sun disc
    float disc = smoothstep(SUN_ANGULAR_COS - 0.00005f, SUN_ANGULAR_COS + 0.00005f, cosAngle);

    // Mie glow halo
    float glow = pow(max(cosAngle, 0.0f), 512.0f) * 0.4f;

    // Transmittance to sun: use top of atmosphere, mu ≈ cosAngle
    float3 T_sun = T_lut(0.0f, cosAngle);

    float3 sunDisc = SUN_SPECTRAL_COLOR * disc * 8.0f;
    float3 glowCol = float3(1.0f, 0.8f, 0.5f) * glow;

    return (sunDisc + glowCol) * T_sun;
}

// ----------------------------------------------------------
// Vertex shader
// ----------------------------------------------------------
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
    if (vertexID == 0u)
        pos = float2(-1.0f, -1.0f);
    else if (vertexID == 1u)
        pos = float2( 3.0f, -1.0f);
    else
        pos = float2(-1.0f,  3.0f);

    output.Pos = float4(pos, 0.0f, 1.0f);
    output.uv  = pos * 0.5f + 0.5f;

    // Reconstruct world-space view ray
    float4 clipPos = float4(pos, 0.5f, 1.0f);
    float4 worldPos4 = mul(pc.invView, mul(pc.invProjection, clipPos));
    output.worldDir = normalize(worldPos4.xyz / worldPos4.w - pc.cameraPos.xyz);

    return output;
}

// ----------------------------------------------------------
// Fragment shader resources
// ----------------------------------------------------------
// Binding layout (set=0):
//   binding=0: Texture2D<float4> tTransmittance (sampled image)
//   binding=1: SamplerState g_Sampler
// The C++ side binds these as two separate descriptors.
[[vk::binding(0, 0)]]
Texture2D<float4> tTransmittance : register(t0);

[[vk::binding(1, 0)]]
SamplerState g_Sampler : register(s0);

// Wrapper that samples using the global sampler
float3 sampleLUT(float2 uv)
{
    return tTransmittance.Sample(g_Sampler, uv).rgb;
}

[[shader("fragment")]]
float4 fragMain(VSOutput input) : SV_Target0
{
    float3 rayDir   = normalize(input.worldDir);
    float3 cameraPos = pc.cameraPos.xyz; // km, planet-centered

    // Reconstruct sun direction from elevation + azimuth
    float sunElevation = pc.sunDir.x;
    float sunAzimuth   = pc.sunDir.y;
    float3 sunDir = normalize(float3(
        cos(sunElevation) * sin(sunAzimuth),
        sin(sunElevation),
        cos(sunElevation) * cos(sunAzimuth)
    ));

    float3 color = float3(0.0f);

    // Only compute scattering if camera is at or above the surface
    float hCam = length(cameraPos) - INNER_RADIUS;
    if (hCam >= -0.1f)
    {
        color = computeSkyRadiance(cameraPos, rayDir, sunDir);
    }

    // Sun disc
    color += renderSun(rayDir, sunDir);

    // Ground below horizon
    if (rayDir.y < 0.0f)
    {
        float t = clamp(-rayDir.y * 10.0f, 0.0f, 1.0f);
        float3 groundColor = float3(0.06f, 0.04f, 0.02f);
        color = lerp(color, groundColor, t);
    }

    // Reinhard tone mapping + gamma correction
    color = color / (color + 1.0f);
    color = pow(color, float3(1.0f / 2.2f));

    return float4(color, 1.0f);
}
