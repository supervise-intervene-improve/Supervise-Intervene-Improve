Shader "Custom/StableInstancedPoint"
{
    Properties
    {
        _BaseColor("Base Color", Color) = (1,1,1,0.95)
        _OverlayAlpha("Overlay Alpha", Range(0, 1)) = 0.95
        _VisibilityGain("Visibility Gain", Range(1, 4)) = 1.75
        _MinVisibleBrightness("Min Visible Brightness", Range(0, 1)) = 0.28
        [HideInInspector] _SrcBlend("Src Blend", Float) = 5
        [HideInInspector] _DstBlend("Dst Blend", Float) = 10
        [HideInInspector] _ZWrite("Z Write", Float) = 0
        [HideInInspector] _ZTest("Z Test", Float) = 8
    }

    SubShader
    {
        Tags { "RenderType"="Transparent" "Queue"="Transparent+100" "RenderPipeline"="UniversalPipeline" }

        Pass
        {
            Blend [_SrcBlend] [_DstBlend]
            Cull Off
            ZWrite [_ZWrite]
            ZTest [_ZTest]

            HLSLPROGRAM
            #pragma target 4.5
            #pragma vertex Vert
            #pragma fragment Frag
            #pragma multi_compile_instancing

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            StructuredBuffer<float4x4> matrixBuffer;
            StructuredBuffer<float4> colorBuffer;

            float4 _BaseColor;
            float _OverlayAlpha;
            float _VisibilityGain;
            float _MinVisibleBrightness;
            float4x4 _RenderPoseMatrix;

            struct Attributes
            {
                float3 positionOS : POSITION;
                uint instanceID : SV_InstanceID;
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float4 color : COLOR;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            Varyings Vert(Attributes input)
            {
                Varyings output;
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);
                uint pointIndex = input.instanceID;

                float4x4 localPointMatrix = matrixBuffer[pointIndex];
                float4 localPointPosition = mul(localPointMatrix, float4(input.positionOS, 1.0));
                float4 worldPos = mul(_RenderPoseMatrix, localPointPosition);
                output.positionCS = TransformWorldToHClip(worldPos.xyz);

                float4 pointColor = saturate(colorBuffer[pointIndex]);
                float3 gainedColor = saturate(pointColor.rgb * _VisibilityGain);
                float maxChannel = max(gainedColor.r, max(gainedColor.g, gainedColor.b));
                float liftAmount = saturate(_MinVisibleBrightness - maxChannel);
                float3 visibleColor = saturate((gainedColor + float3(liftAmount, liftAmount, liftAmount)) * _BaseColor.rgb);
                float alpha = saturate(max(_OverlayAlpha, maxChannel + liftAmount) * _BaseColor.a);
                output.color = float4(visibleColor, alpha);
                return output;
            }

            half4 Frag(Varyings input) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);
                return input.color;
            }
            ENDHLSL
        }
    }
}
