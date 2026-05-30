// scene_frag.slang — Phase 1: Simple forward rendering fragment shader

struct PSInput
{
    float4 position : SV_Position;
    float3 worldPos : TEXCOORD0;
    float3 normal : TEXCOORD1;
    float2 texCoord : TEXCOORD2;
};

// Set 1: separate image + sampler bindings
[[vk::binding(0, 1)]]
Texture2D baseColorTexture;
[[vk::binding(1, 1)]]
SamplerState baseColorSampler;

[shader("fragment")]
float4 fragMain(PSInput input) : SV_Target0
{
    float3 N = normalize(input.normal);

    // Sample albedo
    float3 albedo = baseColorTexture.Sample(baseColorSampler, input.texCoord).rgb;

    // Simple directional light
    float3 lightDir = normalize(float3(0.3, 0.8, 0.5));
    float NdotL = max(dot(N, lightDir), 0.0);

    // Ambient + diffuse
    float3 ambient = float3(0.06, 0.06, 0.08) * albedo;
    float3 diffuse = albedo * NdotL * float3(1.0, 0.95, 0.9);

    // Simple gamma correction
    float3 color = ambient + diffuse;
    color = color / (color + float3(1.0, 1.0, 1.0));             // Reinhard tone mapping
    color = pow(color, float3(1.0 / 2.2, 1.0 / 2.2, 1.0 / 2.2)); // gamma

    return float4(color, 1.0);
}
