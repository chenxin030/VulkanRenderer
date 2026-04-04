// Transmittance LUT precomputation — Bruneton + O'Neil analytic simplification
//
// This compute shader runs once at startup and outputs a 2D transmittance LUT
// indexed by (height, cosine_of_view_angle).
//
// The LUT is used by the sky and aerial-perspective shaders to compute
// T(r, theta) — the transmittance from altitude r along angle theta to the
// top of the atmosphere — in O(1) without per-frame ray marching.
//
// Physical model:
//   - Inner radius (Earth surface):  6360 km
//   - Outer radius (atmosphere top): 6420 km
//   - Rayleigh extinction coefficient (at sea level): 5.8e-3 km^-1
//   - Mie extinction coefficient (at sea level):    2.1e-2 km^-1
//   - Ozone absorption coefficient (at sea level):  1.5e-3 km^-1
//
// References:
//   - Bruneton et al. "A Scalable and Production Ready Sky and Atmosphere Rendering Technique"
//   - O'Neil "Accurate Atmospheric Scattering" (GPU Gems 2, Ch. 16)

struct TransmittanceParams
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
    float2 _padding;
    [[vk::offset(160)]]
    float innerRadius;
    [[vk::offset(164)]]
    float outerRadius;
    [[vk::offset(168)]]
    float RayleighCoeff;
    [[vk::offset(172)]]
    float MieCoeff;
    [[vk::offset(176)]]
    float RayleighScale;
    [[vk::offset(180)]]
    float MieScale;
    [[vk::offset(184)]]
    float Rayleigh4PI;
    [[vk::offset(188)]]
    float Mie4PI;
    [[vk::offset(192)]]
    float RayleighESun;
    [[vk::offset(196)]]
    float MieESun;
    [[vk::offset(200)]]
    float g;
    [[vk::offset(204)]]
    float pad0;
    [[vk::offset(208)]]
    float ozoneCoeff;
    [[vk::offset(212)]]
    float ozoneHeight;
    [[vk::offset(216)]]
    float2 pad1;
};

[[vk::push_constant]]
TransmittanceParams params;

// LUT dimensions (must match C++ side)
static const uint LUT_WIDTH  = 256u;
static const uint LUT_HEIGHT = 64u;

// Atmosphere geometry constants (km)
static const float INNER_RADIUS  = 6360.0f;
static const float OUTER_RADIUS  = 6420.0f;
static const float ATMOSPHERE_HEIGHT = OUTER_RADIUS - INNER_RADIUS; // 60 km

// Linear interpolation: lerp(a, b, t) = a + (b - a) * t
float lerp_float(float a, float b, float t) { return a + (b - a) * t; }

// Density model: exp(-h / H)
float getRayleighDensity(float h)
{
    return exp(-h / params.RayleighScale);
}

float getMieDensity(float h)
{
    return exp(-h / params.MieScale);
}

// Ozone: triangular profile peaking at 25 km
// sigma_O3(h) = sigma_O3_0 * max(0, 1 - |h - 25| / 15)
float getOzoneDensity(float h)
{
    if (params.ozoneCoeff <= 0.0f) return 0.0f;
    return params.ozoneCoeff * max(0.0f, 1.0f - abs(h - params.ozoneHeight) / 15.0f);
}

// Sample density profile at a normalized height h (km above sea level)
float3 sampleDensity(float h)
{
    return float3(
        getRayleighDensity(h),
        getMieDensity(h),
        getOzoneDensity(h)
    );
}

// Compute extinction coefficients (beta_R, beta_M, beta_O3) at normalized height h
float3 computeExtinction(float h)
{
    float3 densities = sampleDensity(h);
    return float3(
        params.RayleighCoeff * densities.x + params.MieCoeff * densities.y + densities.z,
        params.MieCoeff       * densities.y,
        densities.z
    );
}

// Ray-sphere intersection
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

RWTexture2D<float4> transmittanceLUT : register(u0);
[[vk::binding(0, 0)]] // set=0, binding=0

[[shader("compute"), numthreads(8, 8, 1)]]
void compMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if (dispatchThreadID.x >= LUT_WIDTH || dispatchThreadID.y >= LUT_HEIGHT)
        return;

    // ============================================================
    // UV -> (height, cosine_of_view_angle) parameterization
    // ============================================================
    float x_mu  = (float(dispatchThreadID.x) + 0.5f) / float(LUT_WIDTH);
    float x_r   = (float(dispatchThreadID.y) + 0.5f) / float(LUT_HEIGHT);

    // mu: cosine of the view-zenith angle, in [-1, 1]
    float mu  = lerp_float(-1.0f, 1.0f, x_mu);
    // r: distance from planet center, mapped to atmosphere range
    float r   = lerp_float(params.innerRadius, params.outerRadius, x_r);

    // ============================================================
    // Find ray path through atmosphere
    // ============================================================
    float3 pos = float3(0.0f, r, 0.0f);
    float3 dir = float3(0.0f, mu, sqrt(max(0.0f, 1.0f - mu * mu)));

    // Clamp mu to [-1, 1]
    mu = clamp(mu, -1.0f, 1.0f);
    dir.y = mu;
    dir.z = sqrt(max(0.0f, 1.0f - mu * mu));

    // Find where the ray exits the atmosphere
    float outerRadius2 = params.outerRadius * params.outerRadius;
    float b = dot(pos, dir);
    float c = dot(pos, pos) - outerRadius2;
    float disc = b * b - c;
    if (disc < 0.0f)
    {
        transmittanceLUT[int2(dispatchThreadID.x, dispatchThreadID.y)] = float4(1.0f, 1.0f, 1.0f, 1.0f);
        return;
    }
    float far = -b + sqrt(disc);

    if (far <= 0.0f)
    {
        transmittanceLUT[int2(dispatchThreadID.x, dispatchThreadID.y)] = float4(1.0f, 1.0f, 1.0f, 1.0f);
        return;
    }

    // ============================================================
    // Compute optical depth along this ray (16 samples)
    // ============================================================
    float stepSize = far / 16.0f;
    float3 depth = float3(0.0f, 0.0f, 0.0f);

    for (float i = 0.0f; i < 16.0f; i += 1.0f)
    {
        float t = (i + 0.5f) * stepSize;
        float3 s = pos + dir * t;
        float h = max(length(s) - params.innerRadius, 0.0f);
        float3 ext = computeExtinction(h);
        depth += ext * stepSize;
    }

    // ============================================================
    // Apply Beer-Lambert: T = exp(-optical_depth)
    // ============================================================
    float3 transmittance = exp(-depth);

    transmittanceLUT[int2(dispatchThreadID.x, dispatchThreadID.y)] = float4(transmittance, 1.0f);
}
