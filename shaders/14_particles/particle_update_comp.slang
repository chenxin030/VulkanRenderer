struct Particle
{
    float3 position;
    float lifetime;
    float3 velocity;
    float maxLifetime;
    float4 color;
    float size;
    float3 padding;
};

struct ParticleParams
{
    float3 emitterPosition;
    float emissionRate;
    float3 gravity;
    float particleLifetime;
    float size;
    float turbulenceStrength;
    float spread;
    float velocity;
};

[[vk::binding(0, 0)]]
RWStructuredBuffer<Particle> particles;

[[vk::binding(1, 0)]]
ConstantBuffer<ParticleParams> params;

static const float PI = 3.14159265359;
static const uint MAX_PARTICLES = 100000;

float hash(float n)
{
    return fract(sin(n) * 43758.5453123);
}

float3 hash3(float n)
{
    return float3(
        hash(n),
        hash(n + 127.1),
        hash(n + 269.5)
    );
}

float noise(float3 p)
{
    float3 i = floor(p);
    float3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);

    float n = i.x + i.y * 57.0 + i.z * 113.0;
    return mix(
        mix(mix(hash(n), hash(n + 1.0), f.x),
            mix(hash(n + 57.0), hash(n + 58.0), f.x), f.y),
        mix(mix(hash(n + 113.0), hash(n + 114.0), f.x),
            mix(hash(n + 170.0), hash(n + 171.0), f.x), f.y), f.z);
}

float3 noise3D(float3 p)
{
    return float3(
        noise(p),
        noise(p + 100.0),
        noise(p + 200.0)
    );
}

float3 respawnParticle(uint index)
{
    float3 rnd = hash3(float(index) * 123.456);

    float theta = rnd.x * PI * 2.0;
    float phi = rnd.y * params.spread;
    float speed = params.velocity * (0.5 + rnd.z * 0.5);

    float3 dir;
    dir.x = sin(phi) * cos(theta);
    dir.y = cos(phi);
    dir.z = sin(phi) * sin(theta);

    return dir * speed;
}

[numthreads(64, 1, 1)]
[shader("compute")]
void compMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint index = dispatchThreadID.x;
    if (index >= MAX_PARTICLES)
    {
        return;
    }

    Particle p = particles[index];

    float dt = 1.0 / 60.0;

    p.lifetime -= dt;

    if (p.lifetime <= 0.0)
    {
        p.position = params.emitterPosition;
        p.velocity = respawnParticle(index);
        p.lifetime = params.particleLifetime * (0.5 + hash(float(index) * 789.123) * 0.5);
        p.maxLifetime = params.particleLifetime;
        p.size = params.size * (0.8 + hash(float(index) * 456.789) * 0.4);

        float lifeRatio = 1.0 - (p.lifetime / p.maxLifetime);

        float3 youngColor = float3(0.2, 0.4, 1.0);
        float3 midColor = float3(1.0, 0.8, 0.2);
        float3 oldColor = float3(1.0, 0.2, 0.1);

        if (lifeRatio < 0.3)
        {
            p.color = float4(mix(youngColor, midColor, lifeRatio / 0.3), 1.0);
        }
        else
        {
            p.color = float4(mix(midColor, oldColor, (lifeRatio - 0.3) / 0.7), 1.0);
        }

        p.color.a = 1.0;
    }
    else
    {
        float3 turbulence = noise3D(p.position * 0.5 + float3(0.0, float(index) * 0.01, 0.0)) * 2.0 - 1.0;
        turbulence *= params.turbulenceStrength;

        p.velocity += params.gravity * dt;
        p.velocity += turbulence * dt;
        p.position += p.velocity * dt;

        if (p.position.y < -10.0)
        {
            p.lifetime = 0.0;
        }

        float lifeRatio = 1.0 - (p.lifetime / p.maxLifetime);
        p.color.a = 1.0 - lifeRatio * 0.8;
    }

    particles[index] = p;
}
