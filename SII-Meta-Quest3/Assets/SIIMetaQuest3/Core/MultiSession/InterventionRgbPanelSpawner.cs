using UnityEngine;

/// <summary>
/// RGB-panel counterpart to the point-cloud single view. Spawned by
/// InterventionSessionBootstrap.Awake() ONLY when SessionRegistry.SelectedRgbMode
/// is true (the launcher was started with --rgb_mode). Kept in its own file/class
/// — deliberately separate from anything point-cloud-related — so the two single
/// view modes never share code paths that could regress one while changing the
/// other.
///
/// Spawns 4 camera panels (top, left, right, wrist) reusing the exact same
/// SessionThumbnailPanel + ThumbnailRendererBridge construction the multi-session
/// selector grid uses (MultiSessionGridManager.CreatePanelGameObject), but:
///   - laid out in a fixed "diamond" pattern (top / left / right / bottom-wrist)
///     instead of the grid's head-relative spherical math,
///   - true-parented (via an intermediate "RgbDiamondRoot" — see SpawnDiamond) to
///     the SceneAnchorManager-spawned "MujocoScene" GameObject, so repositioning
///     the anchor (Left Menu unlock + nudge + relock, same as the point cloud
///     today) moves and holds the panels exactly the same way,
///   - no hover/select ray wiring — single view doesn't select panels.
/// </summary>
public class InterventionRgbPanelSpawner : MonoBehaviour
{
    [Header("Session Config (set by InterventionSessionBootstrap)")]
    public string publisherIp = "127.0.0.1";
    public int    topicPort   = 7741;

    [Tooltip("Seconds between per-panel 'hb' telemetry lines for the four single-view " +
             "camera panels. These are what make the VR-RGB receive rate measurable in " +
             "adb logcat. 0 disables. Four panels at 1 Hz is a negligible log rate; the " +
             "selector's 15 panels are a different matter and stay off by default.")]
    public float  panelHeartbeatSeconds = 1.0f;

    [Header("Anchor")]
    [Tooltip("Name of the GameObject SceneAnchorManager spawns/owns. Panels are " +
             "parented (via RgbDiamondRoot) to it so they inherit its controller-" +
             "tunable, PlayerPrefs-persisted position/rotation/scale automatically " +
             "via normal Transform parenting (no manual matrix bookkeeping needed, " +
             "unlike the GPU point cloud loader which bypasses Transform for " +
             "instanced draws).")]
    public string anchorName = "MujocoScene";
    [Tooltip("Give up retrying after this many seconds if the anchor never appears.")]
    public float anchorWaitTimeoutSeconds = 10f;

    [Header("Diamond Layout (local space relative to the anchor)")]
    [Tooltip("Horizontal offset (±X) for the left/right panels. Kept small (just over " +
             "half panelWidth) so the two panels sit edge-to-edge instead of spread to " +
             "the sides of the diamond.")]
    public float spreadX = 0.12f;
    [Tooltip("Vertical offset (±Y) for the top/wrist panels.")]
    public float spreadY = 0.22f;
    [Tooltip("Forward offset (+Z, anchor's 'into the scene' direction) for all 4 panels.")]
    public float depthZ = 0.05f;
    [Tooltip("Local yaw applied to every panel so it faces back toward the viewer " +
             "(the anchor's +Z points away from the viewer into the scene). Flip to " +
             "0 if panels appear mirrored/backwards after a build — verify visually.")]
    public float panelFacingYawDeg = 180f;
    [Tooltip("Additional in-plane rotation applied only to the left/right panels. " +
             "MuJoCo's 'left'/'right' cameras were previously only consumed by the " +
             "orientation-agnostic point cloud pipeline (depth pixels are placed in " +
             "3D via the camera extrinsic regardless of 2D array orientation) — viewed " +
             "as flat 2D images for the first time here, they appeared upside down. " +
             "180 is the expected fix for a camera mounted rotated about its own " +
             "optical axis; adjust and rebuild if it under/over-corrects.")]
    public float leftRightCorrectionRotationDeg = 180f;

    [Header("Panel Appearance")]
    [Tooltip("RGB diamond panels only: correct the horizontal mirror introduced by " +
             "panelFacingYawDeg=180, which turns the quad's BACK face toward the viewer " +
             "(visible at all only because Custom/UnlitDoubleSided is Cull Off). The " +
             "selector grid orients its panels with LookRotation(toHead) instead, so it " +
             "shows the FRONT face, needs no correction, and has its own bridge — it is " +
             "not affected by this flag. Applied as a negative localScale.x on the " +
             "Thumbnail quad (see SpawnPanel), NOT as a _MainTex scale/offset.")]
    public bool mirrorPanelImagesHorizontally = true;
    [Tooltip("Which diamond cameras the mirror correction applies to. NOT all four: the " +
             "back-face mirror is geometric and identical for every panel, but 'wrist' " +
             "reads correctly WITHOUT it — its source image arrives already mirrored " +
             "relative to front/left/right (the wrist camera is mounted on the moving " +
             "hand), so the back face was incidentally cancelling that out and reading " +
             "right the whole time. Correcting all four regressed the one panel that was " +
             "already fine. Verified on-device; change only against the headset.")]
    public string[] mirroredCameras = { "front", "left", "right" };
    public float panelWidth  = 0.22f;
    public float panelHeight = 0.165f;
    public Color noSignalColor = new Color(0.45f, 0.45f, 0.50f, 1f);
    [Tooltip("Default border color used when borderMaterial is unassigned (always the " +
             "case here — this component is added via AddComponent at runtime, so " +
             "there is no Inspector to assign borderMaterial from). Without an explicit " +
             "shader-based material the Border quad keeps Unity's default primitive " +
             "material, which renders as solid magenta/purple under URP.")]
    public Color normalBorderColor = new Color(0.25f, 0.25f, 0.25f, 1f);
    public Material panelMaterial;
    public Material borderMaterial;

    [Header("Diamond Tilt (RGB-panel-only control, separate from SceneAnchorManager)")]
    [Tooltip("Degrees/second the diamond array tilts (rotates around its local X axis) " +
             "while PC-adjust mode is unlocked and Right stick X is pushed. Does not " +
             "touch SceneAnchorManager — reads its AdjustModeActive flag only.")]
    public float tiltSpeed = 45f;
    public string playerPrefsKeyTiltRot = "RgbDiamondTilt_Rot";
    public bool persistTiltAcrossSessions = true;

    private bool _spawned;
    private float _waitStartTime;
    private GameObject _diamondRoot;
    private bool _prevAdjustModeActive;
    private bool _hasUnsavedTiltNudge;
    private readonly System.Collections.Generic.List<SessionThumbnailPanel> _panels =
        new System.Collections.Generic.List<SessionThumbnailPanel>();

    void Start()
    {
        _waitStartTime = Time.unscaledTime;
    }

    void Update()
    {
        if (!_spawned)
        {
            var anchorGo = GameObject.Find(anchorName);
            if (anchorGo == null)
            {
                if (Time.unscaledTime - _waitStartTime > anchorWaitTimeoutSeconds)
                {
                    Debug.LogWarning($"[InterventionRgbPanelSpawner] Gave up waiting for '{anchorName}' "
                                    + $"after {anchorWaitTimeoutSeconds:F1}s — RGB panels not spawned.");
                    _spawned = true; // stop polling
                }
                return;
            }

            _spawned = true;
            SpawnDiamond(anchorGo.transform);
            return;
        }

        UpdateTilt();
    }

    private void SpawnDiamond(Transform anchor)
    {
        // The anchor OUTLIVES this component. "MujocoScene" is created with `new GameObject`,
        // so it belongs to the ACTIVE scene (the selector), while this spawner lives in the
        // additively-loaded SIIScene_InterventionV1 -- and parenting moves RgbDiamondRoot into
        // the selector scene too. Exiting single view unloads the intervention scene but leaves
        // BOTH behind, so a plain `new GameObject` here stacked a second diamond (4 more panels,
        // 4 more live ZMQ subscribers, each still pointed at the previously-selected session) on
        // every exit->re-enter cycle. Visually identical, invisibly cumulative -- and the extra
        // sockets inflate Python's peer count, which the sensor-activation heuristic reads.
        var stale = anchor.Find("RgbDiamondRoot");
        if (stale != null)
        {
            Debug.Log("[InterventionRgbPanelSpawner] destroying stale RgbDiamondRoot left by a "
                    + "previous single-view entry (its panels still held ZMQ subscribers).");
            // Rename BEFORE destroying: Destroy is deferred to end of frame, so the doomed
            // object would otherwise still answer a Find("RgbDiamondRoot") made this frame.
            stale.name = "RgbDiamondRoot_destroying";
            Destroy(stale.gameObject);
        }

        // Intermediate transform so the whole 4-panel array can be tilted as a
        // group (Fix 4) without touching SceneAnchorManager or affecting how the
        // panels otherwise inherit the anchor's translate/yaw/roll/scale.
        _diamondRoot = new GameObject("RgbDiamondRoot");
        _diamondRoot.transform.SetParent(anchor, false);
        _diamondRoot.transform.localPosition = Vector3.zero;
        _diamondRoot.transform.localRotation = TryLoadTiltRotation(out var savedTilt) ? savedTilt : Quaternion.identity;

        Quaternion facing = Quaternion.Euler(0f, panelFacingYawDeg, 0f);
        Transform root = _diamondRoot.transform;
        SpawnPanel("front", new Vector3(0f, +spreadY, depthZ), facing, root);
        SpawnPanel("left",  new Vector3(-spreadX, 0f, depthZ), facing, root);
        SpawnPanel("right", new Vector3(+spreadX, 0f, depthZ), facing, root);
        SpawnPanel("wrist", new Vector3(0f, -spreadY, depthZ), facing, root);
        Debug.Log($"[InterventionRgbPanelSpawner] Spawned diamond RGB panels (front/left/right/wrist) "
                + $"anchored to '{anchor.name}' via RgbDiamondRoot, ip={publisherIp}:{topicPort}.");
    }

    // Mirrors MultiSessionGridManager.CreatePanelGameObject's hierarchy (quad +
    // border + label + collider + SessionThumbnailPanel + ThumbnailRendererBridge),
    // copied rather than shared so the working grid-selector code is never touched.
    private void SpawnPanel(string camName, Vector3 localPos, Quaternion facing, Transform parent)
    {
        var root = new GameObject($"RgbPanel_{camName}");
        root.transform.SetParent(parent, false);
        root.transform.localPosition = localPos;
        // "left"/"right" cameras appear upside down when viewed as flat 2D images
        // (see leftRightCorrectionRotationDeg doc above) — apply an extra in-plane
        // twist on top of the shared viewer-facing rotation for those two only.
        Quaternion correction = (camName == "left" || camName == "right")
            ? Quaternion.Euler(0f, 0f, leftRightCorrectionRotationDeg)
            : Quaternion.identity;
        root.transform.localRotation = facing * correction;

        var thumbGo = GameObject.CreatePrimitive(PrimitiveType.Quad);
        thumbGo.name = "Thumbnail";
        thumbGo.transform.SetParent(root.transform, false);
        // Horizontal mirror correction is GEOMETRY (negative X scale), not a _MainTex
        // scale/offset. The texture-transform version only works on shaders that run
        // TRANSFORM_TEX on _MainTex_ST: Custom/UnlitDoubleSided does, but the
        // Sprites/Default fallback below passes IN.texcoord straight through and
        // ignores tiling/offset entirely — so the correction silently did nothing and
        // the panels stayed mirrored with no error anywhere. It could also be dropped
        // by any later material reassignment. Negating the quad's X scale cannot be
        // ignored by a shader and cannot be undone by touching the material; Cull Off
        // makes the reversed winding a non-issue, and the sibling Border/Label quads
        // are untouched. The selector grid builds its own panels and is unaffected.
        //
        // Per-camera, NOT global — see mirroredCameras: 'wrist' must be left alone.
        bool mirrorThis = mirrorPanelImagesHorizontally && IsMirroredCamera(camName);
        thumbGo.transform.localScale = new Vector3(
            mirrorThis ? -panelWidth : panelWidth, panelHeight, 1f);

        Shader guaranteedShader = Shader.Find("Custom/UnlitDoubleSided")
                               ?? Shader.Find("Sprites/Default")
                               ?? Shader.Find("Unlit/Color");

        Material thumbMat;
        if (panelMaterial != null)
        {
            thumbMat = panelMaterial;
        }
        else if (guaranteedShader != null)
        {
            thumbMat = new Material(guaranteedShader);
            thumbMat.color = noSignalColor;
            thumbMat.SetColor("_Color", noSignalColor);
        }
        else
        {
            thumbMat = thumbGo.GetComponent<Renderer>().material;
            thumbMat.color = noSignalColor;
        }
        thumbGo.GetComponent<Renderer>().material = thumbMat;
        Destroy(thumbGo.GetComponent<Collider>());

        var borderGo = GameObject.CreatePrimitive(PrimitiveType.Quad);
        borderGo.name = "Border";
        borderGo.transform.SetParent(root.transform, false);
        borderGo.transform.localScale = new Vector3(panelWidth + 0.012f, panelHeight + 0.012f, 1f);
        borderGo.transform.localPosition = new Vector3(0f, 0f, 0.002f);
        // Always assign an explicit shader-based material (same guaranteed-shader
        // fallback chain as the Thumbnail quad above). Leaving this unset (the old
        // "if (borderMaterial != null)" guard) meant the Border quad kept Unity's
        // default primitive material, which renders as solid magenta/purple under
        // URP — borderMaterial can never be assigned via Inspector here since this
        // whole component is only ever created via AddComponent at runtime.
        Material borderMat;
        if (borderMaterial != null)
        {
            borderMat = borderMaterial;
        }
        else if (guaranteedShader != null)
        {
            borderMat = new Material(guaranteedShader);
            borderMat.color = normalBorderColor;
            borderMat.SetColor("_Color", normalBorderColor);
        }
        else
        {
            borderMat = borderGo.GetComponent<Renderer>().material;
            borderMat.color = normalBorderColor;
        }
        borderGo.GetComponent<Renderer>().material = borderMat;
        Destroy(borderGo.GetComponent<Collider>());

        var canvasGo = new GameObject("Label");
        canvasGo.transform.SetParent(root.transform, false);
        canvasGo.transform.localPosition = new Vector3(0f, -(panelHeight * 0.5f + 0.02f), 0f);
        var canvas = canvasGo.AddComponent<Canvas>();
        canvas.renderMode = RenderMode.WorldSpace;
        var rectTf = canvasGo.GetComponent<RectTransform>();
        rectTf.sizeDelta = new Vector2(panelWidth, 0.05f);
        rectTf.localScale = Vector3.one * 0.003f;
        var textGo = new GameObject("Text");
        textGo.transform.SetParent(canvasGo.transform, false);
        var textRt = textGo.AddComponent<RectTransform>();
        textRt.sizeDelta = new Vector2(panelWidth / 0.003f, 18f);
        var txt = textGo.AddComponent<UnityEngine.UI.Text>();
        txt.fontSize  = 14;
        txt.alignment = TextAnchor.MiddleCenter;
        txt.color     = Color.white;
        txt.text      = camName;

        // Required by SessionThumbnailPanel ([RequireComponent(typeof(Collider))]), but kept
        // DISABLED here: panel.enableSelection=false (set below) means SetVisible never enables
        // it. This matters in combined view — both scenes are loaded, so the grid's
        // right-controller ray would otherwise hit these colliders and Select() a diamond panel,
        // corrupting SessionRegistry and breaking scene switching.
        var col = root.AddComponent<BoxCollider>();
        col.size   = new Vector3(panelWidth, panelHeight, 0.01f);
        col.center = Vector3.zero;

        var panel = root.AddComponent<SessionThumbnailPanel>();
        panel.publisherIp    = publisherIp;
        panel.topicPort      = topicPort;
        panel.thumbnailTopic = $"SimPub/Sensors/{camName}/rgb";
        panel.sessionLabel   = camName;
        panel.borderRenderer    = borderGo.GetComponent<Renderer>();
        panel.thumbnailImage    = null;
        panel.labelText         = txt;
        panel.thumbnailRenderer = thumbGo.GetComponent<Renderer>();
        panel.panelCollider     = col;
        panel.showPlaceholderUntilFrame = true;
        panel.placeholderColor  = noSignalColor;
        panel.enableRiskColoring = false;  // diamond camera panels show a feed, not session risk
        panel.enableLiveIndicator = false; // no LiveIndicator child built for these panels anyway
        panel.enablePausedIndicator = false; // no PausedIndicator child built for these panels anyway
        panel.enableSelection = false;     // NOT selectable — in combined view the grid ray must
                                           // not hover/Select these (would corrupt SessionRegistry
                                           // and break scene switching)
        // Per-panel 1 Hz telemetry. SessionThumbnailPanel defaults this OFF because the
        // SELECTOR spawns 15 panels and 15 lines/s drowned the logcat buffer. Single view
        // spawns four, and without their heartbeat the VR-RGB condition has no measurable
        // receive rate at all -- its `rx` counter existed but only printed on reconnect.
        // So it is enabled here, for these four only, rather than by flipping the default.
        panel.heartbeatLogIntervalSeconds = panelHeartbeatSeconds;

        var bridge = root.AddComponent<ThumbnailRendererBridge>();
        bridge.thumbnailRenderer = thumbGo.GetComponent<Renderer>();
        bridge.panel             = panel;
        // MUST stay false: the mirror is corrected once, in geometry, on the quad's
        // localScale.x above. Setting it here too would flip a second time and cancel out.
        bridge.flipHorizontal    = false;

        // Name the resolved shader and the applied correction, so a future "still
        // mirrored" report is answerable from logcat instead of by re-deriving which of
        // the two mechanisms was live in that particular build (lesson 42: any C# change
        // does nothing until the APK is rebuilt).
        Debug.Log($"[InterventionRgbPanelSpawner] panel '{camName}' shader="
                + $"{(guaranteedShader != null ? guaranteedShader.name : "<none>")} "
                + $"mirrorH={mirrorThis} (geometry, scale.x="
                + $"{thumbGo.transform.localScale.x:F3}) "
                + $"inPlaneRotDeg={(camName == "left" || camName == "right" ? leftRightCorrectionRotationDeg : 0f):F0}");

        _panels.Add(panel);
    }

    /// <summary>True when this camera's panel needs the back-face mirror corrected.
    /// Case-insensitive; an empty/unset mirroredCameras list corrects nothing.</summary>
    private bool IsMirroredCamera(string camName)
    {
        if (mirroredCameras == null) return false;
        foreach (var c in mirroredCameras)
        {
            if (string.Equals(c, camName, System.StringComparison.OrdinalIgnoreCase))
                return true;
        }
        return false;
    }

    /// <summary>Re-point all 4 diamond panels to a different session in place (combined-view
    /// switch, no scene reload). Called by InterventionSessionBootstrap.ReconfigureToSelectedSession.
    /// Each panel drops its stale image and reconnects to the new endpoint; the per-camera topic
    /// strings are unchanged.</summary>
    public void ReconfigureAll(string ip, int newTopicPort)
    {
        publisherIp = ip;
        topicPort   = newTopicPort;
        foreach (var p in _panels)
        {
            if (p != null) p.Reconfigure(ip, newTopicPort);
        }
        Debug.Log($"[InterventionRgbPanelSpawner] ReconfigureAll → {ip}:{newTopicPort} "
                + $"({_panels.Count} diamond panels re-pointed).");
    }

    // -----------------------------------------------------------------------
    // Diamond tilt (RGB-panel-only; SceneAnchorManager itself is never modified)

    private void UpdateTilt()
    {
        if (_diamondRoot == null) return;

        bool adjustActive = SceneAnchorManager.AdjustModeActive;

#if !UNITY_EDITOR
        if (adjustActive)
        {
            // Right stick X is unclaimed during PC-adjust mode (only Right stick Y is
            // used there, for scene scale) and unclaimed by InterventionButtonForwarder's
            // normal-mode bindings — safe to repurpose for diamond tilt. Naturally inert
            // in PC mode since this component (and RgbDiamondRoot) only exists in RGB mode.
            Vector2 rStick = OVRInput.Get(OVRInput.RawAxis2D.RThumbstick);
            if (Mathf.Abs(rStick.x) > 0.1f)
            {
                _diamondRoot.transform.localRotation =
                    Quaternion.AngleAxis(rStick.x * tiltSpeed * Time.deltaTime, Vector3.right)
                    * _diamondRoot.transform.localRotation;
                _hasUnsavedTiltNudge = true;
            }
        }
#endif

        // On the unlock→lock edge (mirrors SceneAnchorManager's own re-lock save),
        // persist the tilt. Pure PlayerPrefs/C# — safe to evaluate in the editor too,
        // it just never fires there since AdjustModeActive only goes true on-device.
        if (_prevAdjustModeActive && !adjustActive && _hasUnsavedTiltNudge && persistTiltAcrossSessions)
        {
            SaveTiltRotation(_diamondRoot.transform.localRotation);
            _hasUnsavedTiltNudge = false;
        }
        _prevAdjustModeActive = adjustActive;
    }

    /// <summary>Persist an unsaved tilt when this component goes away. The only other save site
    /// is the unlock->re-lock edge, so tilting and then exiting single view (which unloads this
    /// scene) discarded the adjustment silently.</summary>
    void OnDisable()      { SavePendingTilt("component disabled / scene unloaded"); }
    void OnApplicationPause(bool paused) { if (paused) SavePendingTilt("application paused"); }
    void OnApplicationQuit()             { SavePendingTilt("application quitting"); }

    private void SavePendingTilt(string reason)
    {
        if (!persistTiltAcrossSessions || !_hasUnsavedTiltNudge || _diamondRoot == null) return;
        SaveTiltRotation(_diamondRoot.transform.localRotation);
        _hasUnsavedTiltNudge = false;
        Debug.Log($"[InterventionRgbPanelSpawner] flushed unsaved diamond tilt ({reason}).");
    }

    private void SaveTiltRotation(Quaternion rot)
    {
        var e = rot.eulerAngles;
        // InvariantCulture -- see PrefsPose. Interpolation used the device culture, which on a
        // comma-decimal locale wrote a string the loader could never parse back.
        PlayerPrefs.SetString(playerPrefsKeyTiltRot, PrefsPose.Format(e));
        PlayerPrefs.Save();
        Debug.Log($"[InterventionRgbPanelSpawner] saved diamond tilt to PlayerPrefs: euler={e}");
    }

    private bool TryLoadTiltRotation(out Quaternion rot)
    {
        rot = Quaternion.identity;
        if (!persistTiltAcrossSessions || !PlayerPrefs.HasKey(playerPrefsKeyTiltRot))
            return false;
        try
        {
            if (!PrefsPose.TryParseVector3(PlayerPrefs.GetString(playerPrefsKeyTiltRot), out var e))
            {
                Debug.LogWarning("[InterventionRgbPanelSpawner] tilt PlayerPrefs unreadable: "
                               + $"'{PlayerPrefs.GetString(playerPrefsKeyTiltRot)}'");
                return false;
            }
            rot = Quaternion.Euler(e.x, e.y, e.z);
            Debug.Log($"[InterventionRgbPanelSpawner] restored diamond tilt from PlayerPrefs: euler={rot.eulerAngles}");
            return true;
        }
        catch (System.Exception ex)
        {
            Debug.LogWarning($"[InterventionRgbPanelSpawner] tilt PlayerPrefs parse failed: {ex.Message}");
            return false;
        }
    }
}
