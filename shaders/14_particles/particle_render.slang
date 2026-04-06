struct SceneUBO
{
    float4x4 projection;
    float4x4 view;
    float3 camPos;
    float pad0;
};

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

[[vk::binding(0, 0)]]
ConstantBuffer<SceneUBO> sceneUbo;

[[vk::binding(1, 0)]]
StructuredBuffer<Particle> particles;

struct VSOutput
{
    float4 Pos : SV_POSITION;
    float4 Color : COLOR0;
    float PointSize : SV_PointSize;
};

[[shader("vertex")]]
VSOutput vertMain(uint vertexID : SV_VertexID)
{
    VSOutput output;

    const Particle p = particles[vertexID];

    const float4 worldPos = float4(p.position, 1.0);
    const float4 viewPos = mul(sceneUbo.view, worldPos);

    output.Pos = mul(sceneUbo.projection, viewPos);
    output.Color = p.color;
    output.PointSize = p.size * 100.0;

    return output;
}

[[shader("fragment")]]
float4 fragMain(VSOutput input) : SV_TARGET
{
    return input.Color;
}
