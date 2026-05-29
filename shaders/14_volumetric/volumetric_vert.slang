// Volumetric Lighting - Vertex Shader (Full-screen triangle)

struct VSOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VSOutput vertMain(uint vertexID: SV_VertexID)
{
    VSOutput output;

    // Single large triangle (matching deferred_lighting pattern)
    float2 pos;
    if (vertexID == 0)
        pos = float2(-1.0, -1.0);
    else if (vertexID == 1)
        pos = float2(3.0, -1.0);
    else
        pos = float2(-1.0, 3.0);

    output.position = float4(pos, 0.0, 1.0);
    output.uv = pos * 0.5 + 0.5;

    return output;
}
