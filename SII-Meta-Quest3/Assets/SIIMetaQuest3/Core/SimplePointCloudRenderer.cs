using UnityEngine;

/// <summary>
/// Simple point cloud renderer using MeshTopology.Points.
/// Requires a material that supports point rendering + vertex colors.
/// </summary>
[RequireComponent(typeof(MeshFilter), typeof(MeshRenderer))]
public class SimplePointCloudRenderer : MonoBehaviour
{
    [Tooltip("Uniform scale applied to incoming points.")]
    public float scale = 1.0f;

    [Tooltip("Shader point size passed to materials that expose _PointSize.")]
    public float pointSize = 1.5f;

    private static readonly int PointSizeId = Shader.PropertyToID("_PointSize");

    private Mesh _mesh;
    private int[] _indices;
    private Vector3[] _scaled;
    private MeshRenderer _meshRenderer;
    private MaterialPropertyBlock _materialProps;

    void Awake()
    {
        var mf = GetComponent<MeshFilter>();
        _mesh = new Mesh();
        _mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
        _mesh.MarkDynamic();
        mf.sharedMesh = _mesh;

        _meshRenderer = GetComponent<MeshRenderer>();
        if (_meshRenderer.sharedMaterial == null)
        {
            Debug.LogWarning("[SimplePointCloudRenderer] No material assigned. Use a point/vertex-color shader.");
        }

        ApplyPointMaterialSettings();
    }

    void OnValidate()
    {
        if (_meshRenderer == null)
            _meshRenderer = GetComponent<MeshRenderer>();
        ApplyPointMaterialSettings();
    }

    private void ApplyPointMaterialSettings()
    {
        if (_meshRenderer == null)
            return;

        if (_meshRenderer.sharedMaterial == null)
            return;

        if (_materialProps == null)
            _materialProps = new MaterialPropertyBlock();

        _meshRenderer.GetPropertyBlock(_materialProps);
        _materialProps.SetFloat(PointSizeId, Mathf.Max(1.0f, pointSize));
        _meshRenderer.SetPropertyBlock(_materialProps);
    }

    public void UpdateCloud(Vector3[] points, Color32[] colors)
    {
        if (points == null || points.Length == 0)
        {
            _mesh.Clear();
            return;
        }

        int count = points.Length;

        if (_indices == null || _indices.Length != count)
        {
            _indices = new int[count];
            for (int i = 0; i < count; i++)
                _indices[i] = i;
        }

        if (_scaled == null || _scaled.Length != count)
            _scaled = new Vector3[count];

        // Apply scale in-place to keep GC down.
        for (int i = 0; i < count; i++)
            _scaled[i] = points[i] * scale;

        _mesh.Clear();
        _mesh.vertices = _scaled;

        if (colors != null && colors.Length == count)
            _mesh.colors32 = colors;

        _mesh.SetIndices(_indices, MeshTopology.Points, 0, false);
        _mesh.RecalculateBounds();
        ApplyPointMaterialSettings();
    }
}
