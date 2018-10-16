using UnityEngine;
using Unity.Burst;
using Unity.Jobs;
using Unity.Collections;
using Unity.Mathematics;
using Rng = Unity.Mathematics.Random;

public class BDCCurve : MonoBehaviour {
	private NativeArray<float2> _points;
	private Rng _rng;

	private void Awake () {
		_points = new NativeArray<float2>(3, Allocator.Persistent);

		_points[0] = new float2(0f, 0f);
        _points[1] = new float2(-1f, 0.5f);
        _points[2] = new float2(0f, 1f);

		_rng = new Rng(1234);
    }

	private void OnDestroy() {
		_points.Dispose();
	}
	
	private void Update () {
		_points[1] += new float2(Input.GetAxis("Horizontal"), Input.GetAxis("Vertical")) * Time.deltaTime;
	}

	private void OnDrawGizmos() {
		if (!Application.isPlaying) {
			return;
		}

		Gizmos.color = Color.blue;
        for (int i = 0; i < _points.Length; i++) {
            Gizmos.DrawSphere(Util.ToVec3(_points[i]), 0.05f);
        }

		Gizmos.color = Color.white;
		float2 pPrev = BDC.Evaluate(_points[0], _points[1], _points[2], 0f);
        Gizmos.DrawSphere(Util.ToVec3(pPrev), 0.01f);
		int steps = 16;
		for (int i = 1; i <= steps; i++) {
			float t = i / (float)steps;
			float2 p = BDC.Evaluate(_points[0], _points[1], _points[2], t);
			Gizmos.DrawLine(Util.ToVec3(pPrev), Util.ToVec3(p));
            Gizmos.DrawSphere(Util.ToVec3(p), 0.01f);
			pPrev = p;
        }
	}

	private void OnGUI() {
		GUILayout.Label("Length: " + BDC.LengthEuclidean(_points[0], _points[1], _points[2], 16));
	}
}

public static class Util {
	public static Vector3 ToVec3(float2 p) {
		return new Vector3(p.x, p.y, 0f);
	}
}

public static class BDC {
	public static float Length(float2 v) {
		return math.sqrt(v.x * v.x + v.y * v.y);
	}

	public static float2 Lerp(float2 a, float2 b, float t) {
		return t * a + (1f - t) * b;
	}
	public static float2 EvaluateWithLerp(float2 a, float2 b, float2 c, float t) {
		return Lerp(Lerp(a, b, t), Lerp(b, c, t), t);
	}

    public static float2 Evaluate(float2 a, float2 b, float2 c, float t) {
        float2 u = 1f - t;
		return u * u * a + 2f * t * u * b + t * t * c;
    }

	public static float LengthEuclidean(float2 a, float2 b, float2 c, int steps) {
		float dist = 0;

        float2 pPrev = BDC.Evaluate(a, b, c, 0f);
        for (int i = 1; i <= steps; i++) {
            float t = i / (float)steps;
            float2 p = BDC.Evaluate(a, b, c, t);
			dist += Length(p - pPrev);
            pPrev = p;
        }

		return dist;
	}
}
