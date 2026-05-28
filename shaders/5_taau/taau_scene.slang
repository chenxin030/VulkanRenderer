struct SceneUBO
{
    float4x4 projection;
    float4x4 view;
    float3 camPos;
};
[[vk::binding(0, 0)]]
ConstantBuffer<SceneUBO> sceneUbo;

struct InstanceData
{
    float4x4 model;
    float4 color;
};
[[vk::binding(1, 0)]]
StructuredBuffer<InstanceData> instanceBuffer;

struct VSInput
{
    float3 inPosition;
    float3 inNormal;
};

struct VSOutput
{
    float4 pos : SV_Position;
    float3 worldPos;
    float3 worldNormal;
    float4 color;
    float4 currClipPos;
    float4 prevClipPos;
};

[shader("vertex")]
VSOutput vertMain(VSInput input, uint instanceID: SV_InstanceID)
{
    VSOutput output;

    float4x4 model = instanceBuffer[instanceID].model;
    float4 worldPos4 = mul(model, float4(input.inPosition, 1.0));
    float3 worldPos = worldPos4.xyz;

    float3x3 normalMat = (float3x3)model;
    float3 worldNormal = normalize(mul(normalMat, input.inNormal));

    float4x4 view = sceneUbo.view;
    float4x4 projection = sceneUbo.projection;

    float4 clipPos = mul(projection, mul(view, float4(worldPos, 1.0)));
    float4 prevClipPos = mul(sceneUbo.projection * sceneUbo.view, float4(worldPos, 1.0));

    output.pos = clipPos;
    output.worldPos = worldPos;
    output.worldNormal = worldNormal;
    output.color = instanceBuffer[instanceID].color;
    output.currClipPos = clipPos;
    output.prevClipPos = prevClipPos;
    return output;
}

[shader("fragment")]
void fragMain(VSOutput input, out float4 color : SV_TARGET0, out float2 velocity : SV_TARGET1)
{
    float3 baseColor = input.color.xyz;
    float3 normal = normalize(input.worldNormal);

    float3 lightDir = normalize(float3(0.5, 1.0, 0.3));
    float NdotL = saturate(dot(normal, lightDir));
    float3 diffuse = baseColor * NdotL * 0.8;

    float3 viewDir = normalize(sceneUbo.camPos - input.worldPos);
    float3 halfVec = normalize(lightDir + viewDir);
    float spec = pow(saturate(dot(normal, halfVec)), 32.0);
    float3 specular = float3(0.2) * spec;

    float3 ambient = baseColor * 0.15;
    color = float4(ambient + diffuse + specular, 1.0);

    float2 currNDC = input.currClipPos.xy / input.currClipPos.w;
    float2 prevNDC = input.prevClipPos.xy / input.prevClipPos.w;
    velocity = currNDC - prevNDC;
}
