Shader "SII/PointCloud/GpuInstancedCubes"
{
    Properties
    {
        _BaseColor ("Tint", Color) = (1,1,1,1)
        _CubeSize ("Cube Size", Float) = 0.003
    }

    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Geometry" "RenderPipeline"="UniversalPipeline" }
        Pass
        {
            Cull Off
            ZWrite On
            ZTest LEqual

            HLSLPROGRAM
            #pragma target 4.5
            #pragma vertex Vert
            #pragma fragment Frag
            #pragma multi_compile_instancing

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            StructuredBuffer<float4> _Positions;
            StructuredBuffer<uint> _PackedColors;

            float4 _BaseColor;
            float _CubeSize;
            float4x4 _RenderPoseMatrix;

            struct Attributes
            {
                float3 positionOS : POSITION;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float4 color : COLOR;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            float4 UnpackColor(uint packed)
            {
                float r = (packed & 0xFFu) / 255.0;
                float g = ((packed >> 8u) & 0xFFu) / 255.0;
                float b = ((packed >> 16u) & 0xFFu) / 255.0;
                float a = ((packed >> 24u) & 0xFFu) / 255.0;
                return float4(r, g, b, a);
            }

            Varyings Vert(Attributes v)
            {
                Varyings o;
                UNITY_SETUP_INSTANCE_ID(v);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);

                #if UNITY_ANY_INSTANCING_ENABLED
                uint pointIndex = unity_InstanceID;
                #else
                uint pointIndex = 0u;
                #endif
                float3 center = _Positions[pointIndex].xyz;
                float3 localPos = (v.positionOS * _CubeSize) + center;
                float4 worldPos = mul(_RenderPoseMatrix, float4(localPos, 1.0));
                o.positionCS = TransformWorldToHClip(worldPos.xyz);
                o.color = UnpackColor(_PackedColors[pointIndex]) * _BaseColor;
                return o;
            }

            half4 Frag(Varyings i) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);
                return i.color;
            }
            ENDHLSL
        }
    }
}
