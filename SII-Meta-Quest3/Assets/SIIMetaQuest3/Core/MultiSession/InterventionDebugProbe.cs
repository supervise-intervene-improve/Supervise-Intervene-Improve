using UnityEngine;

/// <summary>
/// Diagnostic probe spawned by InterventionSessionBootstrap when the user
/// arrives in SIIScene_InterventionV1 from the multi-session selector.
///
/// Creates a small bright-red unlit cube parented under the active camera at
/// a fixed local offset, so the user can visually confirm whether ANYTHING
/// renders in AR passthrough mode. If the cube is visible but the wrist
/// panel / point cloud are not, the camera/passthrough setup is fine and the
/// problem is specific to those subscribers' material or coordinate logic.
///
/// Also emits a once-per-second heartbeat line to make adb logcat parsing
/// trivial.
/// </summary>
public class InterventionDebugProbe : MonoBehaviour
{
    public Vector3 localOffset = new Vector3(0.2f, -0.05f, 0.8f);
    public float cubeSize = 0.1f;
    public Color cubeColor = Color.red;
    public float heartbeatIntervalSeconds = 1.0f;

    private GameObject _cube;
    private Camera _cam;
    private float _nextLogTime;
    private int _frame;

    void Start()
    {
        _cam = Camera.main;
        if (_cam == null)
        {
            Debug.LogWarning("[InterventionDebugProbe] Camera.main is null — cannot spawn debug cube.");
            return;
        }

        _cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
        _cube.name = "InterventionDebugProbe_Cube";
        var collider = _cube.GetComponent<Collider>();
        if (collider != null) Destroy(collider);

        _cube.transform.SetParent(_cam.transform, worldPositionStays: false);
        _cube.transform.localPosition = localOffset;
        _cube.transform.localRotation = Quaternion.identity;
        _cube.transform.localScale    = Vector3.one * Mathf.Max(0.01f, cubeSize);

        Shader sh = Shader.Find("Universal Render Pipeline/Unlit");
        if (sh == null) sh = Shader.Find("Unlit/Color");
        if (sh == null) sh = Shader.Find("Standard");
        var mat = new Material(sh);
        if (mat.HasProperty("_BaseColor")) mat.SetColor("_BaseColor", cubeColor);
        if (mat.HasProperty("_Color"))     mat.SetColor("_Color",     cubeColor);
        var rend = _cube.GetComponent<Renderer>();
        rend.sharedMaterial = mat;
        rend.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        rend.receiveShadows = false;

        Debug.Log(
            $"[InterventionDebugProbe] spawned cube parent='{_cam.name}' "
            + $"localOffset={localOffset} size={cubeSize} shader='{sh?.name}' color={cubeColor}"
        );
    }

    void Update()
    {
        _frame++;
        if (Time.unscaledTime < _nextLogTime) return;
        _nextLogTime = Time.unscaledTime + Mathf.Max(0.1f, heartbeatIntervalSeconds);

        Vector3 camWorldPos = _cam != null ? _cam.transform.position : Vector3.zero;
        Vector3 cubeWorldPos = _cube != null ? _cube.transform.position : Vector3.zero;
        int sessionIdx = SessionRegistry.SelectedSessionIndex;
        Debug.Log(
            $"[InterventionDebugProbe] hb frame={_frame} camWorld={camWorldPos} "
            + $"cubeWorld={cubeWorldPos} session={sessionIdx}"
        );
    }

    void OnDestroy()
    {
        if (_cube != null) Destroy(_cube);
    }
}
