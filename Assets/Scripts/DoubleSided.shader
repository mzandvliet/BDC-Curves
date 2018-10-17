/* 
 * Uses VFACE trick described here: 
 * https://forum.unity.com/threads/double-sided-material.474594/
 * https://forum.unity3d.com/threads/standard-shader-modified-to-be-double-sided-is-very-shiny-on-the-underside.393068/#post-2574717
 */

Shader "Custom/DoubleSided" {
	Properties {
		_Color ("Color", Color) = (1,1,1,1)
		_TopTex ("Top Albedo (RGB)", 2D) = "white" {}
		_BottomTex ("Bottom Albedo (RGB)", 2D) = "white" {}
		_Glossiness ("Smoothness", Range(0,1)) = 0.5
		_Metallic ("Metallic", Range(0,1)) = 0.0
	}
	SubShader {
		Pass
        {
            Tags {"LightMode"="ShadowCaster"}
            ZWrite On
            Cull Off
 
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_shadowcaster
            #include "UnityCG.cginc"
 
            struct v2f {
                V2F_SHADOW_CASTER;
                float2 texcoord : TEXCOORD1;
            };
 
            v2f vert(appdata_base v)
            {
                v2f o;
                TRANSFER_SHADOW_CASTER_NORMALOFFSET(o)
                o.texcoord = v.texcoord;
                return o;
            }
         
            sampler2D _MainTex;
            fixed _Cutoff;
 
            float4 frag(v2f i) : SV_Target
            {
                fixed4 col = tex2D(_MainTex, i.texcoord);
                clip(col.a - _Cutoff);
                SHADOW_CASTER_FRAGMENT(i)
            }
            ENDCG
        }

		Tags { "RenderType"="Opaque" }
		LOD 200
		ZWrite On
		Cull Off

		CGPROGRAM
		// Physically based Standard lighting model, and enable shadows on all light types
		#pragma surface surf Standard fullforwardshadows

		// Use shader model 3.0 target, to get nicer looking lighting
		#pragma target 3.0

		sampler2D _TopTex;
		sampler2D _BottomTex;

		struct Input {
			float2 uv_TopTex;
			fixed facing : VFACE;
		};

		half _Glossiness;
		half _Metallic;
		fixed4 _Color;

		// Add instancing support for this shader. You need to check 'Enable Instancing' on materials that use the shader.
		// See https://docs.unity3d.com/Manual/GPUInstancing.html for more information about instancing.
		// #pragma instancing_options assumeuniformscaling
		UNITY_INSTANCING_BUFFER_START(Props)
			// put more per-instance properties here
		UNITY_INSTANCING_BUFFER_END(Props)

		void surf (Input IN, inout SurfaceOutputStandard o) {
			// Albedo comes from a texture tinted by color
			fixed4 c = 0;
			if (IN.facing < 0.5) {
                c = tex2D (_BottomTex, IN.uv_TopTex) * _Color;
				o.Normal *= -1.0;
			} else {
				c = tex2D (_TopTex, IN.uv_TopTex) * _Color;
			}
			o.Albedo = c.rgb;
			// Metallic and smoothness come from slider variables
			o.Metallic = _Metallic;
			o.Smoothness = _Glossiness;
			o.Alpha = c.a;
		}
		ENDCG
	}
	FallBack "Diffuse"
}
