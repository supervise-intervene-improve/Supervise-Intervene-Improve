using UnityEngine;
using UnityEngine.Rendering;

[DisallowMultipleComponent]
public class GpuInstancedPointCloudRenderer : MonoBehaviour
{
    [Tooltip("Uniform scale applied to incoming points before uploading them to the GPU.")]
    public float scale = 1.0f;

    [Tooltip("Cube edge size in millimeters.")]
    public float pointSize = 8.0f;

    [Tooltip("Optional instanced material. If omitted, uses the attached MeshRenderer material.")]
    public Material instancedMaterial;

    [Tooltip("Loose world-space bounds size used for indirect rendering culling.")]
    public float renderBoundsSize = 1000.0f;

    [Tooltip("Log whenever the point-cloud GPU buffers are resized.")]
    public bool logBufferResize = true;

    private static readonly int PositionsId = Shader.PropertyToID("_Positions");
    private static readonly int PackedColorsId = Shader.PropertyToID("_PackedColors");
    private static readonly int CubeSizeId = Shader.PropertyToID("_CubeSize");
    private static readonly int RenderPoseMatrixId = Shader.PropertyToID("_RenderPoseMatrix");

    private ComputeBuffer _positionBuffer;
    private ComputeBuffer _colorBuffer;
    private ComputeBuffer _argsBuffer;
    private readonly uint[] _argsData = new uint[5];

    private Vector4[] _positionUpload;
    private Color32[] _colorUpload;

    private Mesh _cubeMesh;
    private MaterialPropertyBlock _propertyBlock;
    private MeshRenderer _meshRenderer;
    private Bounds _renderBounds;
    private Matrix4x4 _renderPoseMatrix = Matrix4x4.identity;
    private int _allocatedCapacity;
    private int _declaredCapacity;
    private int _pointCount;
    private bool _loggedRuntimeCapabilities;
    private bool _loggedFirstDrawDiagnostics;
    private bool _runtimeReady = true;
    private string _runtimeBlockReason;

    public bool RuntimeReady => _runtimeReady;
    public string RuntimeBlockReason => _runtimeBlockReason ?? string.Empty;
    public Bounds CurrentRenderBounds => _renderBounds;
    public Matrix4x4 CurrentRenderPoseMatrix => _renderPoseMatrix;

    private void Awake()
    {
        EnsureResources();
    }

    private void EnsureResources()
    {
        if (_propertyBlock == null)
            _propertyBlock = new MaterialPropertyBlock();

        if (_meshRenderer == null)
            _meshRenderer = GetComponent<MeshRenderer>();
        if (_meshRenderer != null)
            _meshRenderer.enabled = false;

        if (instancedMaterial == null && _meshRenderer != null)
            instancedMaterial = _meshRenderer.sharedMaterial;
        if (instancedMaterial != null)
            instancedMaterial.enableInstancing = true;

        if (_cubeMesh == null)
        {
            var temp = GameObject.CreatePrimitive(PrimitiveType.Cube);
            _cubeMesh = temp.GetComponent<MeshFilter>().sharedMesh;
            if (Application.isPlaying)
                Destroy(temp);
            else
                DestroyImmediate(temp);
        }

        if (_renderBounds.size == Vector3.zero)
            _renderBounds = new Bounds(transform.position, Vector3.one * Mathf.Max(1f, renderBoundsSize));
        else
            _renderBounds.size = Vector3.one * Mathf.Max(1f, renderBoundsSize);
        _propertyBlock.SetMatrix(RenderPoseMatrixId, _renderPoseMatrix);

        EvaluateRuntimeSupport();

        if (!_loggedRuntimeCapabilities)
        {
            _loggedRuntimeCapabilities = true;
            string capabilitySummary =
                "[GpuInstancedPointCloudRenderer] "
                + $"api={SystemInfo.graphicsDeviceType} "
                + $"supportsInstancing={SystemInfo.supportsInstancing} "
                + $"supportsComputeShaders={SystemInfo.supportsComputeShaders} "
                + $"material={(instancedMaterial != null ? instancedMaterial.name : "null")} "
                + $"shader={(instancedMaterial != null && instancedMaterial.shader != null ? instancedMaterial.shader.name : "null")} "
                + $"xrStereo={UnityEngine.XR.XRSettings.stereoRenderingMode}";

            if (_runtimeReady)
            {
                Debug.Log(capabilitySummary);
            }
            else
            {
                Debug.LogError($"{capabilitySummary} blocked='{_runtimeBlockReason}'");
            }
        }
    }

    public void SetPointCloud(Vector3[] points, Color32[] colors, int count, int declaredCapacity)
    {
        EnsureResources();
        if (!_runtimeReady)
        {
            Clear();
            return;
        }

        if (points == null || colors == null || count <= 0)
        {
            Clear();
            return;
        }

        int safeCount = Mathf.Min(count, Mathf.Min(points.Length, colors.Length));
        int safeCapacity = Mathf.Max(safeCount, declaredCapacity);
        EnsureCapacity(safeCapacity);

        for (int i = 0; i < safeCount; i++)
        {
            Vector3 p = points[i] * scale;
            _positionUpload[i] = new Vector4(p.x, p.y, p.z, 1f);
            _colorUpload[i] = colors[i];
        }

        _positionBuffer.SetData(_positionUpload, 0, 0, safeCount);
        _colorBuffer.SetData(_colorUpload, 0, 0, safeCount);

        _propertyBlock.SetBuffer(PositionsId, _positionBuffer);
        _propertyBlock.SetBuffer(PackedColorsId, _colorBuffer);
        _propertyBlock.SetFloat(CubeSizeId, Mathf.Max(0.0001f, pointSize * 0.001f * scale));
        _propertyBlock.SetMatrix(RenderPoseMatrixId, _renderPoseMatrix);

        _pointCount = safeCount;
        _declaredCapacity = safeCapacity;
    }

    public void SetRenderPose(Matrix4x4 poseMatrix, Vector3 boundsCenter)
    {
        EnsureResources();
        _renderPoseMatrix = poseMatrix;
        _renderBounds.center = boundsCenter;
        _renderBounds.size = Vector3.one * Mathf.Max(1f, renderBoundsSize);
        _propertyBlock.SetMatrix(RenderPoseMatrixId, _renderPoseMatrix);
    }

    public void Clear()
    {
        _pointCount = 0;
    }

    private void EnsureCapacity(int declaredCapacity)
    {
        if (!_runtimeReady)
            return;

        declaredCapacity = Mathf.Max(1, declaredCapacity);
        if (_allocatedCapacity == declaredCapacity
            && _positionBuffer != null
            && _colorBuffer != null
            && _argsBuffer != null)
            return;

        ReleaseBuffers();

        _allocatedCapacity = declaredCapacity;
        _positionUpload = new Vector4[_allocatedCapacity];
        _colorUpload = new Color32[_allocatedCapacity];

        _positionBuffer = new ComputeBuffer(_allocatedCapacity, sizeof(float) * 4);
        _colorBuffer = new ComputeBuffer(_allocatedCapacity, sizeof(uint));
        _argsBuffer = new ComputeBuffer(1, _argsData.Length * sizeof(uint), ComputeBufferType.IndirectArguments);

        if (logBufferResize)
        {
            Debug.Log(
                $"[GpuInstancedPointCloudRenderer] Allocated GPU buffers capacity={_allocatedCapacity} on '{name}'."
            );
        }
    }

    private void Update()
    {
        if (!_runtimeReady)
            return;

        if (_pointCount <= 0 || instancedMaterial == null || _cubeMesh == null || _argsBuffer == null)
            return;

        _renderBounds.size = Vector3.one * Mathf.Max(1f, renderBoundsSize);
        _propertyBlock.SetMatrix(RenderPoseMatrixId, _renderPoseMatrix);

        _argsData[0] = _cubeMesh != null ? _cubeMesh.GetIndexCount(0) : 0u;
        _argsData[1] = (uint)_pointCount;
        _argsData[2] = _cubeMesh != null ? _cubeMesh.GetIndexStart(0) : 0u;
        _argsData[3] = _cubeMesh != null ? _cubeMesh.GetBaseVertex(0) : 0u;
        _argsData[4] = 0u;
        _argsBuffer.SetData(_argsData);

        Graphics.DrawMeshInstancedIndirect(
            _cubeMesh,
            0,
            instancedMaterial,
            _renderBounds,
            _argsBuffer,
            0,
            _propertyBlock
        );

        if (!_loggedFirstDrawDiagnostics)
        {
            _loggedFirstDrawDiagnostics = true;
            Debug.Log(
                "[GpuInstancedPointCloudRenderer] First draw "
                + $"name='{name}' declaredCapacity={_declaredCapacity} points={_pointCount} "
                + $"boundsCenter={_renderBounds.center} boundsSize={_renderBounds.size} "
                + $"poseMatrix={FormatMatrix(_renderPoseMatrix)}"
            );
        }
    }

    private void OnDisable()
    {
        Clear();
    }

    private void OnDestroy()
    {
        ReleaseBuffers();
    }

    private void ReleaseBuffers()
    {
        try { _positionBuffer?.Release(); } catch { }
        try { _colorBuffer?.Release(); } catch { }
        try { _argsBuffer?.Release(); } catch { }
        _positionBuffer = null;
        _colorBuffer = null;
        _argsBuffer = null;
        _allocatedCapacity = 0;
    }

    private void EvaluateRuntimeSupport()
    {
        _runtimeReady = true;
        _runtimeBlockReason = null;

        if (Application.platform == RuntimePlatform.Android
            && SystemInfo.graphicsDeviceType != GraphicsDeviceType.Vulkan)
        {
            _runtimeReady = false;
            _runtimeBlockReason =
                "Quest GPU point clouds require Vulkan on Android, but the runtime graphics API is "
                + $"{SystemInfo.graphicsDeviceType}. Rebuild with Android Graphics APIs set to Vulkan only.";
            return;
        }

        if (!SystemInfo.supportsInstancing)
        {
            _runtimeReady = false;
            _runtimeBlockReason = "GPU point-cloud rendering requires instancing support on the current runtime.";
        }
    }

    private static string FormatMatrix(Matrix4x4 matrix)
    {
        return
            $"[[{matrix.m00:F3},{matrix.m01:F3},{matrix.m02:F3},{matrix.m03:F3}],"
            + $"[{matrix.m10:F3},{matrix.m11:F3},{matrix.m12:F3},{matrix.m13:F3}],"
            + $"[{matrix.m20:F3},{matrix.m21:F3},{matrix.m22:F3},{matrix.m23:F3}],"
            + $"[{matrix.m30:F3},{matrix.m31:F3},{matrix.m32:F3},{matrix.m33:F3}]]";
    }
}
// This component renders large point clouds using GPU instancing with an indirect draw call.