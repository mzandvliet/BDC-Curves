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
		_curve = new NativeArray<float3>(4, Allocator.Persistent);
        _distanceCache = new NativeArray<float>(32, Allocator.Persistent);

		_curve[0] = new float3(-2f, 3f, 0f);
        _curve[1] = new float3(-1f, 0.5f, 0f);
        _curve[2] = new float3(1f, 0f, 0f);
        _curve[3] = new float3(2f, 1f, 0f);

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
		// _curve[0] += new float3(Input.GetAxis("Horizontal"), Input.GetAxis("Vertical"), 0f) * Time.deltaTime * 10f;

		// Relax();

		BDC3Cube.CacheDistances(_curve, _distanceCache);

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

    // Todo: this needs to be rewritten with 2 control points in the middle for the cubic
	private void Relax() {
		// First step: apply control point constraints

		_curve[0] = ClipHeight(_curve[0], 0.05f, 2f);

		var handleDelta = _curve[0] - _curve[2];
		if (math.length(handleDelta) > 4f) {
			_curve[0] = _curve[2] + math.normalize(handleDelta) * 4f;
		}

		// Step 2, relax the curvature control to be above ground, and keep paper area constant

		int iters = 0;
		float lengthError = 1f;
		while (lengthError > 0.001f && iters < 64) {
			int refinementSteps = 4 + iters / 16;
            float length = BDC3Cube.LengthEuclidApprox(_curve, refinementSteps);
			float3 midPoint = (_curve[2] + _curve[0]) * 0.5f;
            float3 cord = math.normalize(_curve[2] - _curve[0]);
			float3 perp = math.cross(cord, new float3(0f, 0f, 1f));
            lengthError = length - 4f;

            _curve[1] += perp * (lengthError < 0f ? (lengthError * lengthError) * 1f : 0);
            _curve[1] += math.normalize((_curve[1] - midPoint)) * -lengthError * 1f;
			_curve[1] += new float3(0f, -0.02f * math.abs(lengthError), 0f); // a gravity term

			_curve[1] = ClipHeight(_curve[1], 0f, 4f);

			iters++;
		}
	}

	private static float3 ClipHeight(float3 p, float min, float max) {
		return new float3(p.x, math.clamp(p.y, min, max), p.z);
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
		float3 pPrev = BDC3Cube.Evaluate(_curve, 0f);
        Gizmos.DrawSphere(pPrev, 0.01f);
		int steps = 16;
		for (int i = 1; i <= steps; i++) {
			float t = (i / (float)steps);
			float3 p = BDC3Cube.Evaluate(_curve, t);
			Gizmos.DrawLine(pPrev, p);
            Gizmos.DrawSphere(p, 0.01f);
			pPrev = p;
        }
	}

	private void OnGUI() {
		GUILayout.Label("Length: " + BDC3Cube.LengthEuclidApprox(_curve, 16));
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

				var p = BDC3Cube.Evaluate(curve, gridPosUnit.x);
				p.z = gridPosUnit.z * 4f;

				normals[i] = BDC3Cube.EvaluateNormal(curve, gridPosUnit.x, new float3(0,1,0));
                verts[i] = p;

                // Todo: this calculation can be cumulative along mesh, this is pretty wasteful
                float uvx = BDC3Cube.LengthEuclidApprox(distances, gridPosUnit.x) / distances[distances.Length-1];
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

// Borrowing from: https://www.youtube.com/watch?v=o9RK6O2kOKo
public static class BDC3Cube {
    public static float3 Lerp(float3 a, float3 b, float t) {
        return t * a + (1f - t) * b;
    }
    public static float3 EvaluateCasteljau(NativeArray<float3> p, float t) {
        float3 bc = Lerp(p[1],p[2],t);
        return Lerp(Lerp(Lerp(p[0], p[1], t), bc, t), Lerp(bc, Lerp(p[2], p[3], t), t), t);
    }

    public static float3 Evaluate(NativeArray<float3> p, float t) {
        float omt = 1f - t;
        float omt2 = omt * omt;
        float t2 = t * t;
        return 
            p[0] * (omt2 * omt) + 
            p[1] * (3f * omt2 * t) +
            p[2] * (3f * omt * t2) +
            p[3] * (t2 * t);
    }

    public static float3 EvaluateTangent(NativeArray<float3> c, float t) {
        float omt = 1f - t;
        float omt2 = omt * omt;
        float t2 = t * t;
        return  math.normalize(
            c[0] * (-omt2) +
            c[1] * (3f * omt2 - 2f * omt) +
            c[2] * (-3f * t2 + 2f * t) +
            c[3] * (t2)
        );
    }

    public static float3 EvaluateNormal(NativeArray<float3> c, float t, float3 up) {
        float3 tangent = EvaluateTangent(c, t);
        float3 binorm = math.cross(up, tangent);
        return math.normalize(math.cross(tangent, binorm));
    }

    public static float LengthEuclidApprox(NativeArray<float3> c, int steps) {
        float dist = 0;

        float3 pPrev = c[0];
        for (int i = 1; i <= steps; i++) {
            float t = i / (float)steps;
            float3 p = Evaluate(c, t);
            dist += math.length(p - pPrev);
            pPrev = p;
        }

        return dist;
    }

    public static float LengthEuclidApprox(NativeArray<float3> c, int steps, float t) {
        float dist = 0;

        float3 pPrev = c[0];
        for (int i = 1; i <= steps; i++) {
            float tNow = t * (i / (float)steps);
            float3 p = Evaluate(c, tNow);
            dist += math.length(p - pPrev);
            pPrev = p;
        }

        return dist;
    }

    public static float LengthEuclidApprox(NativeArray<float> distances, float t) {
        t = t * (float)(distances.Length - 1);
        int ti = (int)math.floor(t);
        if (ti >= distances.Length - 1) {
            return distances[distances.Length - 1];
        }
        return math.lerp(distances[ti], distances[ti + 1], t - (float)ti);
    }

    // Instead of storing at linear t spacing, why not store with non-linear t-spacing and lerp between them
    public static void CacheDistances(NativeArray<float3> c, NativeArray<float> outDistances) {
        float dist = 0;
        outDistances[0] = 0f;
        float3 pPrev = c[0];
        for (int i = 1; i < outDistances.Length; i++) {
            float t = i / (float)(outDistances.Length - 1);
            float3 p = Evaluate(c, t);
            dist += math.length(p - pPrev);
            outDistances[i] = dist;
            pPrev = p;
        }
    }
}

public static class BDC3Quad {
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

        float3 pPrev = Evaluate(a, b, c, 0f);
        for (int i = 1; i <= steps; i++) {
            float t = i / (float)steps;
            float3 p = Evaluate(a, b, c, t);
            dist += Length(p - pPrev);
            pPrev = p;
        }

        return dist;
    }

    public static float LengthEuclidApprox(float3 a, float3 b, float3 c, int steps, float t) {
        float dist = 0;

        float3 pPrev = Evaluate(a, b, c, 0f);
        for (int i = 1; i <= steps; i++) {
            float tNow = t * (i / (float)steps);
            float3 p = Evaluate(a, b, c, tNow);
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
        float3 pPrev = Evaluate(a, b, c, 0f); // Todo: this is just point a
		for (int i = 1; i < outDistances.Length; i++) {
            float t = i / (float)(outDistances.Length-1);
            float3 p = Evaluate(a, b, c, t);
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