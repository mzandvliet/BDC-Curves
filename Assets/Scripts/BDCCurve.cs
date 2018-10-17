using UnityEngine;
using Unity.Burst;
using Unity.Jobs;
using Unity.Collections;
using Unity.Mathematics;
using Rng = Unity.Mathematics.Random;
using Unity.Collections.LowLevel.Unsafe;

/* Todo:

	- Arrange mesh edges such that there is always a middle one that ends up right at the crease
	(without this, with low tesselation, you see the crease jitter back and forth as edges slide in and out)

	A few more approaches for mesh deformation to try: 
	- Could also make it a softbody sim with constraints. Would fix uv-swimming.
	- Compute shader

	Reformulate without Euclidean distance metric.

	Cubic spline
	Momentum for control points. Now, given that we have a no-stretch constraint,
	the paper should not have the degrees of freedom that allow the control points
	to go places where scale goes non-unit. So any momentum we give it will need
	to play out in the subset of unitary state space. It feels a little bit like
	rotation. Momentum can rotate away.
 */

public class BDCCurve : MonoBehaviour {
	[SerializeField] private Material _mat;

	private NativeArray<float3> _curve;
	private NativeArray<float> _distanceCache;
	private Rng _rng;

    private NativeArray<float3> _verts;
    private NativeArray<float3> _normals;
    private NativeArray<int> _triangles;
    private NativeArray<float2> _uvs;

    private Vector3[] _vertsMan;
    private Vector3[] _normalsMan;
    private int[] _trianglesMan;
    private Vector2[] _uvsMan;

	private Mesh _mesh;
	private MeshRenderer _renderer;
	private MeshFilter _meshFilter;

    const int RES = 64 + 1;
    const int NUMVERTS = RES * RES;
	const int NUMTRIS = (RES - 1) * (RES - 1) * 6;

	JobHandle _handle;


	private void Awake () {
		_curve = new NativeArray<float3>(3, Allocator.Persistent);
        _distanceCache = new NativeArray<float>(64, Allocator.Persistent);

		_curve[0] = new float3(0f, 0f, 0f);
        _curve[1] = new float3(2f, 0f, 0f);
        _curve[2] = new float3(4f, 0f, 0f);

		_rng = new Rng(1234);

		_mesh = new Mesh();
		_meshFilter = gameObject.AddComponent<MeshFilter>();
		_renderer = gameObject.AddComponent<MeshRenderer>();
		_meshFilter.mesh = _mesh;
		_renderer.material = _mat;

        _verts = new NativeArray<float3>(NUMVERTS, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        _normals = new NativeArray<float3>(NUMVERTS, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        _triangles = new NativeArray<int>(NUMTRIS, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        _uvs = new NativeArray<float2>(NUMVERTS, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);

        _vertsMan = new Vector3[NUMVERTS];
		_normalsMan = new Vector3[NUMVERTS];
        _trianglesMan = new int[NUMTRIS];
        _uvsMan = new Vector2[NUMVERTS];
    }

	private void OnDestroy() {
		_curve.Dispose();
		_distanceCache.Dispose();

		_verts.Dispose();
		_normals.Dispose();
		_triangles.Dispose();
		_uvs.Dispose();
	}
	
	private void Update () {
		_curve[0] += new float3(Input.GetAxis("Horizontal"), Input.GetAxis("Vertical"), 0f) * Time.deltaTime * 4f;

		Relax();

		BDC3.CacheDistances(_curve[0], _curve[1], _curve[2], _distanceCache);

        var j = new MakeMeshJob();
        j.verts = _verts;
        j.normals = _normals;
        j.triangles = _triangles;
        j.uvs = _uvs;
        j.curve = _curve;
		j.distances = _distanceCache;
        _handle = j.Schedule();
		JobHandle.ScheduleBatchedJobs();
	}

	private void Relax() {
		// Todo: I want the partial derivatives of the following with respect to _pos[1]
		// But I can cheat for now

		// First step: apply control point constraints

		_curve[0] = ClipHeight(_curve[0], 0.05f);

		var handleDelta = _curve[0] - _curve[2];
		if (BDC3.Length(handleDelta) > 4f) {
			_curve[0] = _curve[2] + math.normalize(handleDelta) * 4f;
		}

		// Step 2, relax the curvature control to be above ground, and keep paper area constant
		// Todo: momentum

		int iters = 0;
		float deltaLength = 1f;
		while (deltaLength > 0.001f && iters < 64) {
			int refinementSteps = 4 + iters / 16;
            float length = BDC3.LengthEuclidApprox(_curve[0], _curve[1], _curve[2], refinementSteps);
            float3 midPoint = (_curve[0] + _curve[2]) * 0.5f;
            float3 deltaFromMid = _curve[1] - midPoint;
            deltaLength = 4f - length;
            _curve[1] += deltaFromMid * deltaLength * Time.deltaTime * 20f;
			_curve[1] += new float3(0f, -0.5f * Time.deltaTime, 0f); // a gravity term

			_curve[1] = ClipHeight(_curve[1], 0f);
            _curve[1] = StayLeftOf(_curve[1], _curve[2].x);

			iters++;
		}
	}

	private static float3 ClipHeight(float3 p, float min) {
		return new float3(p.x, math.max(min, p.y), p.z);
	}

    private static float3 StayLeftOf(float3 p, float goal) {
        return new float3(math.min(goal, p.x), p.y, p.z);
    }

	private void LateUpdate() {
        UpdateMesh();
	}

	private void OnDrawGizmos() {
		if (!Application.isPlaying) {
			return;
		}

		Gizmos.color = Color.blue;
        for (int i = 0; i < _curve.Length; i++) {
            Gizmos.DrawSphere(_curve[i], 0.05f);
        }

		Gizmos.color = Color.white;
		float3 pPrev = BDC3.Evaluate(_curve[0], _curve[1], _curve[2], 0f);
        Gizmos.DrawSphere(pPrev, 0.01f);
		int steps = 16;
		for (int i = 1; i <= steps; i++) {
			float t = (i / (float)steps);
			float3 p = BDC3.Evaluate(_curve[0], _curve[1], _curve[2], t);
			Gizmos.DrawLine(pPrev, p);
            Gizmos.DrawSphere(p, 0.01f);
			pPrev = p;
        }
	}

	private void OnGUI() {
		GUILayout.Label("Length: " + BDC3.LengthEuclidApprox(_curve[0], _curve[1], _curve[2], 16));
	}

	private void UpdateMesh() {
		_handle.Complete();

		Util.Copy(_vertsMan, _verts);
        Util.Copy(_normalsMan, _normals);
        Util.Copy(_trianglesMan, _triangles);
        Util.Copy(_uvsMan, _uvs);

		// The below is now the biggest bottleneck, at 0.24ms per frame on my machine
        _mesh.vertices = _vertsMan;
        _mesh.normals = _normalsMan;
        _mesh.triangles = _trianglesMan;
        _mesh.uv = _uvsMan;
        _mesh.UploadMeshData(false);
	}

    [BurstCompile]
    private struct MakeMeshJob : IJob {
        public NativeArray<float3> verts;
        public NativeArray<int> triangles;
        public NativeArray<float2> uvs;
        public NativeArray<float3> normals;

        public NativeArray<float3> curve;
		public NativeArray<float> distances;

        public void Execute() {
            for (int i = 0; i < verts.Length; i++) {
                var gridPos = Math.ToXZFloat(i, RES);
				var gridPosUnit = gridPos / (float)(RES - 1);

				var p = BDC3.Evaluate(curve[0], curve[1], curve[2], gridPosUnit.x);
				p.z = gridPosUnit.z * 4f;

				normals[i] = BDC3.EvaluateNormalApprox(curve[0], curve[1], curve[2], gridPosUnit.x);
                verts[i] = p;

                // Todo: this calculation can be cumulative along mesh, this is pretty wasteful
                float uvx = BDC3.LengthEuclidApprox(distances, gridPosUnit.x) / distances[distances.Length-1];
                uvs[i] = new float2(uvx, gridPosUnit.z);
				
            }

			int idx = 0;
            for (int y = 0; y < RES-1; y++) {
				for (int x = 0; x < RES-1; x++) {
                    triangles[idx++] = Math.ToIndex(x, y, RES);
                    triangles[idx++] = Math.ToIndex(x, y+1, RES);
                    triangles[idx++] = Math.ToIndex(x+1, y+1, RES);

                    triangles[idx++] = Math.ToIndex(x+1, y+1, RES);
                    triangles[idx++] = Math.ToIndex(x+1, y, RES);
                    triangles[idx++] = Math.ToIndex(x, y, RES);
            	}
			}
        }
    }
}



public static class Math {
    public const float Tau = 6.2831853071795864769f;
    public const float Pi = Tau / 2f;

    public static float2 ToXYFloat(int idx, int2 dimensions) {
        return new float2(
            idx % dimensions.x,
            idx / dimensions.x
        );
    }

    public static float3 ToXZFloat(int idx, int2 dimensions) {
        return new float3(
            idx % dimensions.x,
			0f,
            idx / dimensions.x
        );
    }

    public static int2 ToXY(int idx, int2 dimensions) {
        return new int2(
            idx % dimensions.x,
            idx / dimensions.x
        );
    }

    public static int ToIndex(int x, int y, int res) {
        return y * res + x;
    }
}

public static class Util {
	public static Vector3 ToVec3(float2 p) {
		return new Vector3(p.x, p.y, 0f);
	}

    public static unsafe void Copy(Vector3[] destination, NativeArray<float3> source) {
        fixed (void* vertexArrayPointer = destination) {
            UnsafeUtility.MemCpy(
				vertexArrayPointer,
				NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(source),
				destination.Length * (long)UnsafeUtility.SizeOf<float3>());
        }
    }

    public static unsafe void Copy(Vector2[] destination, NativeArray<float2> source) {
        fixed (void* vertexArrayPointer = destination) {
            UnsafeUtility.MemCpy(
                vertexArrayPointer,
                NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(source),
                destination.Length * (long)UnsafeUtility.SizeOf<float2>());
        }
    }

    public static unsafe void Copy(int[] destination, NativeArray<int> source) {
        fixed (void* vertexArrayPointer = destination) {
            UnsafeUtility.MemCpy(
                vertexArrayPointer,
                NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(source),
                destination.Length * (long)UnsafeUtility.SizeOf<int>());
        }
    }

}

public static class BDC3 {
    public static float Length(float3 v) {
        return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    }

    public static float3 Lerp(float3 a, float3 b, float t) {
        return t * a + (1f - t) * b;
    }
    public static float3 EvaluateCasteljau(float3 a, float3 b, float3 c, float t) {
        return Lerp(Lerp(a, b, t), Lerp(b, c, t), t);
    }

    public static float3 Evaluate(float3 a, float3 b, float3 c, float t) {
        float3 u = 1f - t;
        return u * u * a + 2f * t * u * b + t * t * c;
    }

    public static float3 EvaluateNormalApprox(float3 a, float3 b, float3 c, float t) {
		const float EPS = 0.001f;

        float3 p0 = Evaluate(a,b,c,t-EPS);
        float3 p1 = Evaluate(a,b,c,t+EPS);

		return math.cross(new float3(0,0,1), math.normalize(p1 - p0));
    }

    public static float LengthEuclidApprox(float3 a, float3 b, float3 c, int steps) {
        float dist = 0;

        float3 pPrev = BDC3.Evaluate(a, b, c, 0f);
        for (int i = 1; i <= steps; i++) {
            float t = i / (float)steps;
            float3 p = BDC3.Evaluate(a, b, c, t);
            dist += Length(p - pPrev);
            pPrev = p;
        }

        return dist;
    }

    public static float LengthEuclidApprox(float3 a, float3 b, float3 c, int steps, float t) {
        float dist = 0;

        float3 pPrev = BDC3.Evaluate(a, b, c, 0f);
        for (int i = 1; i <= steps; i++) {
            float tNow = t * (i / (float)steps);
            float3 p = BDC3.Evaluate(a, b, c, tNow);
            dist += Length(p - pPrev);
            pPrev = p;
        }

        return dist;
    }

    public static float LengthEuclidApprox(NativeArray<float> distances, float t) {
        t = t * (float)(distances.Length-1);
		int ti = (int)math.floor(t);
		if (ti >= distances.Length-1) {
			return distances[distances.Length-1];
		}
		return math.lerp(distances[ti], distances[ti+1], t - (float)ti);
    }

	// Instead of storing at linear t spacing, why not store with non-linear t-spacing and lerp between them
	public static void CacheDistances(float3 a, float3 b, float3 c, NativeArray<float> outDistances) {
        float dist = 0;
		outDistances[0] = 0f;
        float3 pPrev = BDC3.Evaluate(a, b, c, 0f); // Todo: this is just point a
		for (int i = 1; i < outDistances.Length; i++) {
            float t = i / (float)(outDistances.Length-1);
            float3 p = BDC3.Evaluate(a, b, c, t);
            dist += Length(p - pPrev);
			outDistances[i] = dist;
            pPrev = p;
		}
	}
}


public static class BDC2 {
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

        float2 pPrev = BDC2.Evaluate(a, b, c, 0f);
        for (int i = 1; i <= steps; i++) {
            float t = i / (float)steps;
            float2 p = BDC2.Evaluate(a, b, c, t);
            dist += Length(p - pPrev);
            pPrev = p;
        }

        return dist;
    }
}