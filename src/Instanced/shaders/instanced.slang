struct GlobalUBO {
    float4x4 view;
    float4x4 proj;
};
ConstantBuffer<GlobalUBO> globalUbo;

struct InstanceData {
    float4x4 model;
};
StructuredBuffer<InstanceData> instanceBuffer;

struct VSInput {
    float3 inPosition;
    float3 inColor;
    float2 inTexCoord;
};

struct VSOutput {
    float4 pos : SV_Position;
    float3 fragColor;
    float2 fragTexCoord;
};

[shader("vertex")]
VSOutput vertMain(VSInput input, uint instanceID : SV_InstanceID) {
    VSOutput output;
    float4x4 model = instanceBuffer[instanceID].model;
    output.pos = mul(globalUbo.proj, mul(globalUbo.view, mul(model, float4(input.inPosition, 1.0))));
    output.fragColor = input.inColor;
    output.fragTexCoord = input.inTexCoord;
    return output;
}

Sampler2D texture;

[shader("fragment")]
float4 fragMain(VSOutput vertIn) : SV_TARGET {
   return texture.Sample(vertIn.fragTexCoord);
}
