// Volumetric Lighting - Scene Fragment Shader
// Renders scene and outputs view-space position in color buffer for volumetric effect

struct FSInput
{
    float4 position : SV_Position;
    float4 viewPos : TEXCOORD0;
    float3 worldPos : TEXCOORD1;
    float3 normal : TEXCOORD2;
    float4 color : TEXCOORD3;
};

[shader("fragment")]
void fragMain(FSInput input, out float4 color : SV_TARGET0)
{
    // Output view position in RGB, depth in A
    float depth = -input.viewPos.z;
    color = float4(input.viewPos.xyz, depth);
}
