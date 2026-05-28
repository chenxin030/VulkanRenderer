struct PushConsts
{
    float2 scale;
    float2 translate;
};
[[vk::push_constant]]
PushConsts pc;

[[vk::binding(0, 0)]]
Sampler2D fontTexture;

struct VSInput
{
    float2 Pos;
    float2 UV;
    float4 Color;
};

struct VSOutput
{
    float4 Pos : SV_POSITION;
    float2 UV;
    float4 Color;
};

[shader("vertex")]
VSOutput vertMain(VSInput input)
{
    VSOutput output;
    output.Pos = float4(input.Pos * pc.scale + pc.translate, 0.0, 1.0);
    output.UV = input.UV;
    output.Color = input.Color;
    return output;
}

[shader("fragment")]
float4 fragMain(VSOutput input) : SV_TARGET
{
    float4 tex = fontTexture.Sample(input.UV);
    return input.Color * tex;
}
