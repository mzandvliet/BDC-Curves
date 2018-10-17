using UnityEngine;
using Unity.Burst;
using Unity.Jobs;
using Unity.Collections;
using Unity.Mathematics;
using Rng = Unity.Mathematics.Random;
using Unity.Collections.LowLevel.Unsafe;

/* Todo:
	A few approaches for mesh deformation: 
	
	Regenerate mesh each frame, or do displacement in vertex shader
	Vertex shader approach would only require a single spline to define the whole thing.
	Could also store mesh in compute buffer and use DrawProcedural
	Using Burst for this test
	

	Need to calculate derivatives for normals
	Need to move paper by control point (which itself is constrained)
	Need to apply constant-area constraint and make middle control point solve that constraint
	
	When I have that, I send the mail.

	So set it up that we move the paper edge, and it is constraint to a euclidean max distance from the start
	of the fold. Then use a numerical algorithm to move the middle control point to satisfy length == 1

	Reformulate without Euclidean distance metric.
 */

public class BDCCurve : MonoBehaviour {
	[SerializeField] private Material _mat;

	private NativeArray<float3> _points;
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

	JobHandle _handle;


	private void Awake () {
		_points = new NativeArray<float3>(3, Allocator.Persistent);

		_points[0] = new float3(0f, 2f, 0f);
        _points[1] = new float3(0f, 0f, 0f);
        _points[2] = new float3(4f, 0f, 0f);

		_rng = new Rng(1234);

		_mesh = new Mesh();
		_meshFilter = gameObject.AddComponent<MeshFilter>();
		_renderer = gameObject.AddComponent<MeshRenderer>();
		_meshFilter.mesh = _mesh;
		_renderer.material = _mat;

        _verts = new NativeArray<float3>(NUMVERTS, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        _normals = new NativeArray<float3>(NUMVERTS, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        _triangles = new NativeArray<int>((RES-1)*(RES-1)*6, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        _uvs = new NativeArray<float2>(NUMVERTS, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);

        _vertsMan = new Vector3[NUMVERTS];
		_normalsMan = new Vector3[NUMVERTS];
        _trianglesMan = new int[(RES - 1) * (RES - 1) * 6];
        _uvsMan = new Vector2[NUMVERTS];
    }

	private void OnDestroy() {
		_points.Dispose();

		_verts.Dispose();
		_normals.Dispose();
		_triangles.Dispose();
		_uvs.Dispose();
	}
	
	private void Update () {
		_points[1] += new float3(Input.GetAxis("Horizontal"), Input.GetAxis("Vertical"), 0f) * Time.deltaTime;

        var j = new MakeMeshJob();
        j.verts = _verts;
        j.normals = _normals;
        j.triangles = _triangles;
        j.uvs = _uvs;
        j.points = _points;
        _handle = j.Schedule();
		JobHandle.ScheduleBatchedJobs();
	}

	private void LateUpdate() {
        UpdateMesh();
	}

	private void OnDrawGizmos() {
		if (!Application.isPlaying) {
			return;
		}

		Gizmos.color = Color.blue;
        for (int i = 0; i < _points.Length; i++) {
            Gizmos.DrawSphere(_points[i], 0.05f);
        }

		Gizmos.color = Color.white;
		float3 pPrev = BDC3.Evaluate(_points[0], _points[1], _points[2], 0f);
        Gizmos.DrawSphere(pPrev, 0.01f);
		int steps = 16;
		for (int i = 1; i <= steps; i++) {
			float t = (i / (float)steps);
			float3 p = BDC3.Evaluate(_points[0], _points[1], _points[2], t);
			Gizmos.DrawLine(pPrev, p);
            Gizmos.DrawSphere(p, 0.01f);
			pPrev = p;
        }
	}

	private void OnGUI() {
		GUILayout.Label("Length: " + BDC3.LengthEuclidApprox(_points[0], _points[1], _points[2], 16));
	}

	private void UpdateMesh() {
		_handle.Complete();

		Util.Copy(_vertsMan, _verts);
        Util.Copy(_normalsMan, _normals);
        Util.Copy(_trianglesMan, _triangles);
        Util.Copy(_uvsMan, _uvs);

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

        public NativeArray<float3> points;

        public void Execute() {
            for (int i = 0; i < verts.Length; i++) {
                var pos = Math.ToXZFloat(i, RES);
				var posNorm = new float2(pos.x / (float)(RES-1), pos.z / (float)(RES-1));

				var p = BDC3.Evaluate(points[0], points[1], points[2], posNorm.x);
				p.z = pos.z / (float)(RES - 1) * 4f;

				normals[i] = BDC3.EvaluateNormalApprox(points[0], points[1], points[2], posNorm.x);
                verts[i] = p;
				uvs[i] = posNorm;
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

    public static unsafe NativeArray<float3> GetNativeVertexArrays(Vector3[] vertexArray, NativeArray<float3> verts) {
        fixed (void* vertexBufferPointer = vertexArray) {
            UnsafeUtility.MemCpy(NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(verts),
                vertexBufferPointer, vertexArray.Length * (long)UnsafeUtility.SizeOf<float3>());
        }

        return verts;
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
}
