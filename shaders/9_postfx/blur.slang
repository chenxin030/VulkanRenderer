// Separable Gaussian blur shader
// Used for both horizontal and vertical passes of bloom blur
// Uses 9-tap Gaussian kernel (sigma = 2.0)
//
// The blur direction is determined by which pipeline uses this shader:
//   blurHPipeline: texelSize = float2(4.0/1920, 0) - blur horizontally
//   blurVPipeline: texelSize = float2(0, 4.0/1080) - blur vertically

[[vk::binding(0, 0)]] Sampler2D<float4> inputTexture;

struct VSOutput
{
    float4 Pos : SV_POSITION;
    float2 uv;
};

[[shader("vertex")]]
VSOutput vertMain(uint vertexID : SV_VertexID)
{
    VSOutput output;
    float2 pos;
    if (vertexID == 0) pos = float2(-1.0, -1.0);
    else if (vertexID == 1) pos = float2(3.0, -1.0);
    else pos = float2(-1.0, 3.0);
    output.Pos = float4(pos, 0.0, 1.0);
    output.uv = pos * 0.5 + 0.5;
    return output;
}

// 9-tap Gaussian weights (sigma = 2.0)
static const float GAUSSIAN_WEIGHTS[9] = {
    0.0162162162f, 0.0540540541f, 0.1216216216f,
    0.1945945946f, 0.2270270270f,
    0.1945945946f, 0.1216216216f, 0.0540540541f, 0.0162162162f
};

struct BlurPushConstants
{
    float2 texelSize; // direction & magnitude: (x,0) = H, (0,y) = V
    float padding0;
    float padding1;
    float padding2;
};
[[vk::push_constant]] BlurPushConstants pc;

[[shader("fragment")]]
float4 fragMain(VSOutput input) : SV_Target0
{
    float3 color = float3(0.0f);

    float2 texelSize = pc.texelSize;

    for (int i = -4; i <= 4; ++i)
    {
        float2 offset = float(i) * texelSize;
        float3 sample = inputTexture.Sample(input.uv + offset).rgb;
        color += sample * GAUSSIAN_WEIGHTS[i + 4];
    }

    return float4(color, 1.0f);
}
