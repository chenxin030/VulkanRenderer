// ssao_blur.slang — Phase 4: SSAO bilateral blur pass for vkrEngine
// Full-screen triangle, reads SSAO result, outputs blurred AO

struct BlurUBO
{
    float2 texelSize;
};
[[vk::binding(0, 0)]]
ConstantBuffer<BlurUBO> blurUbo;

[[vk::binding(1, 0)]]
Sampler2D<float> ssaoInput;

struct VSOutput
{
    float4 Pos : SV_POSITION;
    float2 uv : TEXCOORD0;
};

static const int KERNEL_SIZE = 4;

[shader("vertex")]
VSOutput vertMain(uint vertexID: SV_VertexID)
{
    VSOutput output;

    float2 pos;
    if (vertexID == 0u)
        pos = float2(-1.0, -1.0);
    else if (vertexID == 1u)
        pos = float2(3.0, -1.0);
    else
        pos = float2(-1.0, 3.0);

    output.Pos = float4(pos, 0.0, 1.0);
    output.uv = pos * 0.5 + 0.5;
    return output;
}

[shader("fragment")]
float4 fragMain(VSOutput input) : SV_Target0
{
    float2 texelSize = blurUbo.texelSize;

    float result = 0.0;
    float weights = 0.0;

    for (int x = -KERNEL_SIZE; x <= KERNEL_SIZE; ++x)
    {
        for (int y = -KERNEL_SIZE; y <= KERNEL_SIZE; ++y)
        {
            float2 offset = float2(float(x), float(y)) * texelSize;
            float sampleAO = ssaoInput.Sample(input.uv + offset);

            float weight = 1.0 / (1.0 + length(float2(x, y)));
            result += sampleAO * weight;
            weights += weight;
        }
    }

    float ao = result / max(weights, 1e-6);
    return float4(ao, ao, ao, 1.0);
}
