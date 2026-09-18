#if UNITY_EDITOR
using UnityEngine;
using UnityEngine.SceneManagement;

/// <summary>
/// Editor-only play-mode helper for testing the session selector without a Quest.
///
/// The core problem: OVR/XR renders a stereo barrel-distorted view directly to the
/// Game window. Simply setting targetTexture=null is not enough — the XR subsystem
/// overrides it. This script disables ALL existing cameras and creates an independent
/// flat camera that the Game window will use instead.
///
/// Controls:
///   WASD / QE   = move
///   Mouse       = look (while cursor is locked)
///   Left-click  = select the panel you're aiming at
///   Esc         = unlock cursor | click = re-lock
/// </summary>
[DefaultExecutionOrder(-500)]   // run BEFORE GridManager so camera exists when grid spawns
public class EditorTestHelper : MonoBehaviour
{
    [Header("Navigation")]
    public float moveSpeed  = 2.5f;
    public float lookSpeed  = 80.0f;

    [Header("Starting position")]
    [Tooltip("Where the editor camera spawns. Matches OVR CenterEyeAnchor default so the "
           + "grid appears at the same relative distance as on device.")]
    public Vector3 startPosition = new Vector3(0f, 0f, 2.23f);
    public Vector3 startEuler    = Vector3.zero;

    [Header("Testing")]
    [Tooltip("If true, load SIIScene_InterventionV1 when a panel is selected. "
           + "Set false to just log the selection without switching scenes.")]
    public bool   loadSceneOnSelect    = true;
    public string interventionSceneName = "SIIScene_InterventionV1";

    // -----------------------------------------------------------------------

    private Camera   _debugCam;
    private Transform _cam;
    private Vector2  _rot;
    private bool     _locked;

    // Exposed so MultiSessionGridManager can use this ray for hover detection.
    [HideInInspector] public bool    hasEditorRay;
    [HideInInspector] public Vector3 editorRayOrigin;
    [HideInInspector] public Vector3 editorRayDirection;

    // -----------------------------------------------------------------------

    void Awake()
    {
        if (!Application.isEditor) { enabled = false; return; }

        // --- Step 1: disable every camera in the scene so OVR stops rendering ---
        foreach (var c in Object.FindObjectsByType<Camera>(FindObjectsSortMode.None))
        {
            if (c != null)
            {
                c.enabled     = false;
                c.targetTexture = null;     // release any VR render texture
            }
        }

        // --- Step 2: create a clean, independent camera ---
        var camGo = new GameObject("_EditorDebugCamera");
        camGo.tag = "MainCamera";
        _debugCam = camGo.AddComponent<Camera>();

        _debugCam.clearFlags      = CameraClearFlags.SolidColor;
        _debugCam.backgroundColor = new Color(0.08f, 0.08f, 0.10f, 1f);
        _debugCam.nearClipPlane   = 0.01f;
        _debugCam.farClipPlane    = 60f;
        _debugCam.fieldOfView     = 70f;
        _debugCam.depth           = 99;
        _debugCam.targetTexture   = null;       // explicit: render to screen

        _cam = camGo.transform;
        _cam.position = startPosition;
        _cam.eulerAngles = startEuler;

        _rot.y = startEuler.y;
        _rot.x = startEuler.x;

        // Cursor lock
        LockCursor(true);

        Debug.Log("[EditorTestHelper] Clean editor camera created at " + startPosition
                + "\n  WASD/QE=move  |  Mouse=look  |  LMB=select panel  |  Esc=unlock");
    }

    void Update()
    {
        if (_cam == null) return;

        // --- Cursor toggle ---
        if (_locked && Input.GetKeyDown(KeyCode.Escape))
            LockCursor(false);
        else if (!_locked && Input.GetMouseButtonDown(0))
        {
            LockCursor(true);
            return;
        }

        if (_locked)
        {
            HandleMovement();
            HandleLook();
        }

        UpdateEditorRay();

        if (_locked && Input.GetMouseButtonDown(0))
            FireSelect();
    }

    // -----------------------------------------------------------------------

    private void HandleMovement()
    {
        float h  = Input.GetAxis("Horizontal");
        float v  = Input.GetAxis("Vertical");
        float up = (Input.GetKey(KeyCode.E) ? 1f : 0f)
                 - (Input.GetKey(KeyCode.Q) ? 1f : 0f);

        _cam.position += (_cam.right   * h
                        + _cam.forward * v
                        + Vector3.up   * up)
                        * moveSpeed * Time.deltaTime;
    }

    private void HandleLook()
    {
        _rot.y += Input.GetAxis("Mouse X") * lookSpeed * Time.deltaTime;
        _rot.x -= Input.GetAxis("Mouse Y") * lookSpeed * Time.deltaTime;
        _rot.x  = Mathf.Clamp(_rot.x, -80f, 80f);
        _cam.rotation = Quaternion.Euler(_rot.x, _rot.y, 0f);
    }

    private void UpdateEditorRay()
    {
        hasEditorRay       = (_cam != null);
        editorRayOrigin    = _cam != null ? _cam.position    : Vector3.zero;
        editorRayDirection = _cam != null ? _cam.forward     : Vector3.forward;

        // Cyan ray in Scene view — shows exactly where you're aiming
        if (_cam != null)
            Debug.DrawRay(editorRayOrigin, editorRayDirection * 6f, Color.cyan);
    }

    private void FireSelect()
    {
        if (_cam == null) return;

        if (!Physics.Raycast(editorRayOrigin, editorRayDirection,
                             out RaycastHit hit, 8f))
        {
            Debug.Log("[EditorTestHelper] Click — no panel hit.");
            return;
        }

        var panel = hit.collider.GetComponentInParent<SessionThumbnailPanel>();
        if (panel == null)
        {
            Debug.Log($"[EditorTestHelper] Hit '{hit.collider.name}' — not a panel.");
            return;
        }

        Debug.Log($"[EditorTestHelper] Selected session {panel.sessionIndex} "
                + $"port={panel.topicPort}  label='{panel.sessionLabel}'");

        panel.Select();

        Debug.Log($"[EditorTestHelper] SessionRegistry → idx={SessionRegistry.SelectedSessionIndex} "
                + $"ip={SessionRegistry.SelectedPublisherIp} port={SessionRegistry.SelectedTopicPort}");

        if (loadSceneOnSelect)
        {
            Debug.Log($"[EditorTestHelper] Loading {interventionSceneName} ...");
            SceneManager.LoadScene(interventionSceneName);
        }
    }

    private void LockCursor(bool locked)
    {
        _locked          = locked;
        Cursor.lockState = locked ? CursorLockMode.Locked : CursorLockMode.None;
        Cursor.visible   = !locked;
    }

    void OnDestroy()
    {
        LockCursor(false);
    }
}
#endif
