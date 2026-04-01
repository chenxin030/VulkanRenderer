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
    float Size : PSIZE;
    float2 UV : TEXCOORD0;
};

[[shader("vertex")]]
VSOutput vertMain(uint vertexID : SV_VertexID)
{
    VSOutput output;

    Particle p = particles[vertexID];

    float4 worldPos = float4(p.position, 1.0);
    float4 viewPos = mul(sceneUbo.view, worldPos);
    float4 clipPos = mul(sceneUbo.projection, viewPos);

    output.Pos = clipPos;
    output.Color = p.color;
    output.Size = p.size * 100.0;
    output.UV = float2(0.0, 0.0);

    return output;
}

[[shader("fragment")]]
float4 fragMain(VSOutput input) : SV_TARGET
{
    float2 center = float2(0.5, 0.5);

    float2 pixelCoord = float2(
        (input.Pos.x),
        (input.Pos.y)
    );

    float2 toCenter = pixelCoord - center;
    float dist = length(toCenter);

    float radius = 0.5;
    float softEdge = 0.1;

    float alpha = 1.0 - smoothstep(radius - softEdge, radius + softEdge, dist);

    float4 color = input.Color;
    color.a *= alpha;

    if (color.a < 0.01)
    {
        discard;
    }

    return color;
}
