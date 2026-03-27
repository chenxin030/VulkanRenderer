struct VSOut
{
    float4 pos : SV_Position;
    float3 col : COLOR0;
};

[shader("vertex")]
VSOut vertMain(uint vertexId : SV_VertexID)
{
    VSOut o;

    float2 pos;
    pos.x = (vertexId == 2) ? 3.0 : -1.0;
    pos.y = (vertexId == 1) ? -3.0 : 1.0;

    o.pos = float4(pos, 0.0, 1.0);

    // Simple RGB for visibility
    if (vertexId == 0) o.col = float3(1.0, 0.2, 0.2);
    if (vertexId == 1) o.col = float3(0.2, 1.0, 0.2);
    if (vertexId == 2) o.col = float3(0.2, 0.4, 1.0);
    return o;
}

[shader("fragment")]
float4 fragMain(VSOut i) : SV_TARGET
{
    return float4(i.col, 1.0);
}

