using System;
using System.Collections.Generic;
using System.Threading;
using NetMQ;
using NetMQ.Sockets;
using UnityEngine;
using UnityEngine.Rendering;

/// <summary>
/// Head-relative horizontal bar chart displayed in SIIScene_InterventionV1.
/// Shows the raw ACC risk score (0–1) for each running session in a vertical stack.
///
/// Each bar subscribes to SimPub/Status/risk on its session's topic port via a
/// lightweight NetMQ SubscriberSocket. Bars grow left-to-right; color transitions
/// green → yellow → red as risk increases.
///
/// Right-controller ray hover + right index trigger → navigate directly to that
/// session's InterventionV1 (calls InterventionBackButton.NavigateTo).
///
/// Wiring: InterventionSessionBootstrap calls SetDiscoveryData() in Awake(), which
/// builds the chart GameObjects and opens subscriber sockets. Must be called before
/// Start() so sockets are ready on the first frame.
///
/// Teardown: InterventionBackButton.StopSceneSubscribers() calls StopNetMq() before
/// any scene transition so all pollers are stopped while the NetMQ context is alive.
/// </summary>
[DefaultExecutionOrder(-8800)]
public class RiskBarChart : MonoBehaviour
{
    [Header("Chart Layout")]
    public float   maxBarWidth      = 0.32f;   // meters, risk=1 → full width (wider = easier ray select)
    public float   barHeight        = 0.038f;  // meters per bar
    public float   barGap           = 0.012f;  // meters between bars
    public float   labelOffsetX     = -0.055f; // label sits to the left of the bar track
    public float   fontSize         = 0.012f;  // TextMesh characterSize
    public float   titleOffsetY     = 0.028f;  // extra gap between the top bar and the "RISK" title

    [Header("Default Placement (camera-relative, used only if no saved pose)")]
    [Tooltip("X = right, Y = up, Z = forward relative to the camera at scene load.")]
    public Vector3 defaultLocalOffset = new Vector3(0.37f, 0.20f, 0.75f);

    [Header("Interaction")]
    public float   hoverHighlightAlpha = 0.9f;
    public float   armingDelaySeconds  = 1.5f; // prevent trigger-fire on scene entry
    [Tooltip("Hovered bar scales up by this factor and pushes toward the viewer (like the selector grid).")]
    public float   hoverScaleMultiplier = 1.12f;
    [Tooltip("Meters the hovered bar pushes toward the viewer (local -Z of the chart).")]
    public float   hoverForwardPush     = 0.02f;
    [Tooltip("Max ray length for right-controller bar hover/select (meters).")]
    public float   rayMaxDistance       = 5.0f;
    [Tooltip("Name of the TrackingSpace transform (parent of the controller anchors). The "
           + "right-controller pose is TrackingSpace-local and must be transformed to world "
           + "space to hit the world-fixed bar colliders.")]
    public string  trackingSpaceName    = "TrackingSpace";

    [Header("Placement Mode (L3 = left-stick click to toggle)")]
    [Tooltip("Default LOCKED so the controller can't fight interaction. Click L3 to unlock "
           + "for nudging, L3 again to relock + save.")]
    public bool  controlsLocked   = true;
    public float translationSpeed = 0.2f;   // m/s on each axis
    public float rotationSpeed    = 30.0f;  // deg/s on each axis

    [Header("Persistence")]
    public string playerPrefsKeyPos = "RiskChart_Pos";
    public string playerPrefsKeyRot = "RiskChart_Rot";
    public bool   persistPoseAcrossSessions = true;

    // -----------------------------------------------------------------------
    // Static state consumed by InterventionBackButton

    /// <summary>Index of the bar the right-controller ray is currently hovering (-1 if none).</summary>
    public static int HoveredBarIndex { get; private set; } = -1;

    /// <summary>True while the chart is UNLOCKED (placement mode). InterventionButtonForwarder,
    /// InterventionBackButton and SceneAnchorManager read this to suppress their own controls so
    /// the adjust inputs (A/B/X/Y, sticks, L-triggers) don't double-fire.</summary>
    public static bool AdjustModeActive { get; private set; }

    // -----------------------------------------------------------------------
    // Internal

    private string _ip;
    private int    _baseTopicPort;
    private int    _portStep;
    private int    _nSessions;

    private struct BarEntry
    {
        public GameObject Root;
        public Transform  FillTransform;   // scaled on X to show risk
        public Renderer   FillRenderer;
        public Renderer   TrackRenderer;
        public BoxCollider Collider;
        public int         SessionIndex;
        public int         TopicPort;
        public Vector3     BaseLocalPos;   // Root local pos at rest (for hover restore)
        public Vector3     BaseLocalScale; // Root local scale at rest (for hover restore)
    }

    private BarEntry[]         _bars;
    private int[]              _riskRaw;   // float bits, thread-safe via Interlocked
    private SubscriberSocket[] _subs;
    private NetMQPoller[]      _pollers;
    private volatile bool      _shuttingDown;

    private GameObject _chartRoot;

    private float _armingDeadline;
    private bool  _armed;

    // Placement-mode state (mirror of SceneAnchorManager).
    private bool _lockButtonWasPressed;
    private bool _hasUnsavedNudges;

    private Transform _trackingSpace;

    // -----------------------------------------------------------------------
    // Public API called by InterventionSessionBootstrap

    /// <summary>Builds the chart. Must be called once before Start() (called from Bootstrap Awake).</summary>
    public void SetDiscoveryData(string ip, int baseTopicPort, int portStep, int nSessions)
    {
        _ip            = ip;
        _baseTopicPort = baseTopicPort;
        _portStep      = portStep;
        _nSessions     = Mathf.Max(0, nSessions);
    }

    /// <summary>Stop all NetMQ pollers — called by InterventionBackButton before any scene load.</summary>
    public void StopNetMq()
    {
        _shuttingDown = true;
        if (_pollers == null) return;
        for (int i = 0; i < _pollers.Length; i++)
            ShutdownOne(i);
    }

    // -----------------------------------------------------------------------

    void Start()
    {
        _armingDeadline = Time.unscaledTime + Mathf.Max(0f, armingDelaySeconds);
        _armed = false;
        HoveredBarIndex = -1;
        AdjustModeActive = !controlsLocked;

        if (_nSessions <= 0) return;

        BuildChart();
        PlaceChart();
        OpenSubscribers();
        Debug.Log($"[RiskBarChart] built {_nSessions} bars, subscribed to SimPub/Status/risk "
                + $"on {_ip} ports {_baseTopicPort}..{_baseTopicPort + (_nSessions - 1) * _portStep}");
    }

    /// <summary>World-place the chart once: PlayerPrefs pose if saved, else camera-relative
    /// fallback (right-and-up, facing the viewer). Kept top-level so it stays world-fixed
    /// (no per-frame head re-anchor) and can be nudged in placement mode.</summary>
    private void PlaceChart()
    {
        if (_chartRoot == null) return;
        _chartRoot.transform.SetParent(null, false);

        if (persistPoseAcrossSessions && TryLoadPose(out var savedPos, out var savedRot))
        {
            _chartRoot.transform.SetPositionAndRotation(savedPos, savedRot);
            Debug.Log($"[RiskBarChart] restored pose from PlayerPrefs: pos={savedPos} rot={savedRot.eulerAngles}");
            return;
        }

        var cam = Camera.main;
        if (cam != null)
        {
            Vector3 pos = cam.transform.position
                        + cam.transform.forward * defaultLocalOffset.z
                        + cam.transform.up      * defaultLocalOffset.y
                        + cam.transform.right   * defaultLocalOffset.x;
            // Face the viewer so the bars read head-on.
            Quaternion rot = Quaternion.LookRotation((pos - cam.transform.position).normalized, Vector3.up);
            _chartRoot.transform.SetPositionAndRotation(pos, rot);
            Debug.Log($"[RiskBarChart] placed at default camera-relative pos={pos}.");
        }
        else
        {
            Debug.LogWarning("[RiskBarChart] Camera.main is null — leaving chart at origin.");
        }
    }

    void LateUpdate()
    {
        if (_bars == null || _bars.Length == 0) return;

        // Update bar fills from latest risk values (works in editor too).
        for (int i = 0; i < _bars.Length; i++)
        {
            float risk = System.BitConverter.ToSingle(
                System.BitConverter.GetBytes(
                    Interlocked.CompareExchange(ref _riskRaw[i], 0, 0)), 0);
            UpdateBar(i, risk);
        }

#if !UNITY_EDITOR
        // ----- Placement mode (L3 toggle + nudge) -----
        HandlePlacementMode();

        // While EITHER adjust mode is active, the chart is being positioned (or the PC
        // anchor is) — suppress ray interaction so face buttons/sticks don't select.
        if (AdjustModeActive || SceneAnchorManager.AdjustModeActive)
        {
            HoveredBarIndex = -1;
            for (int i = 0; i < _bars.Length; i++)
            {
                SetTrackHighlight(i, false);
                SetBarHovered(i, false);
            }
            return;
        }

        // ----- Ray hover detection (world-space ray; bars are world-fixed) -----
        int hovered = -1;
        if (TryGetControllerRay(out Vector3 rayOrigin, out Vector3 rayDir))
        {
            Ray ray = new Ray(rayOrigin, rayDir);
            for (int i = 0; i < _bars.Length; i++)
            {
                if (_bars[i].Collider != null
                    && _bars[i].Collider.bounds.IntersectRay(ray, out float dist)
                    && dist <= rayMaxDistance)
                {
                    hovered = i;
                    break;
                }
            }
        }
        HoveredBarIndex = hovered;

        // Highlight + bring hovered bar forward (selector-grid style).
        for (int i = 0; i < _bars.Length; i++)
        {
            SetTrackHighlight(i, i == hovered);
            SetBarHovered(i, i == hovered);
        }

        // Arming delay + held-release gate.
        if (Time.unscaledTime < _armingDeadline) return;
        if (!_armed)
        {
            if (OVRInput.Get(OVRInput.RawButton.RIndexTrigger)) return;
            _armed = true;
        }

        // MC condition: the right controller belongs to the arm for the duration of an
        // intervention, so a clutch pull that happens to cross a risk bar must not jump to
        // another session mid-takeover. Same rule as the grid's select-on-trigger; the
        // operator regains navigation by accepting (Left X) or rejecting (Left grip).
        if (MotionControllerModeManager.McControlsLocked) return;

        // Navigate on trigger press while hovering a bar (not the current session).
        if (hovered >= 0
            && hovered != SessionRegistry.SelectedSessionIndex
            && OVRInput.GetDown(OVRInput.RawButton.RIndexTrigger))
        {
            var back = FindAnyObjectByType<InterventionBackButton>();
            if (back != null)
                back.NavigateTo(hovered, _ip, _baseTopicPort, _portStep);
        }
#endif
    }

#if !UNITY_EDITOR
    /// <summary>Port of SceneAnchorManager's controller tuning, keyed off L3 instead of
    /// Left Menu. Roll moves to R3 only (L3 is now the toggle). Ignores the toggle while
    /// the PC anchor is being adjusted so the two modes stay mutually exclusive.</summary>
    private void HandlePlacementMode()
    {
        if (_chartRoot == null) return;

        // ----- Lock toggle (L3 / left-stick click) -----
        bool lockPressed = OVRInput.Get(OVRInput.RawButton.LThumbstick);
        if (lockPressed && !_lockButtonWasPressed && !SceneAnchorManager.AdjustModeActive)
        {
            controlsLocked = !controlsLocked;
            AdjustModeActive = !controlsLocked;
            Debug.Log($"[RiskBarChart] Locked = {controlsLocked} (AdjustModeActive={AdjustModeActive})");
            if (controlsLocked && _hasUnsavedNudges && persistPoseAcrossSessions)
            {
                SavePose(_chartRoot.transform.position, _chartRoot.transform.rotation);
                _hasUnsavedNudges = false;
            }
        }
        _lockButtonWasPressed = lockPressed;

        if (controlsLocked) return;

        var t = _chartRoot.transform;

        // ----- Translation (left stick XY, L-triggers Z) -----
        Vector3 move = Vector3.zero;
        Vector2 leftStick = OVRInput.Get(OVRInput.RawAxis2D.LThumbstick);
        move.x += leftStick.x;
        move.y += leftStick.y;
        if (OVRInput.Get(OVRInput.RawButton.LIndexTrigger)) move.z += 1.0f;
        if (OVRInput.Get(OVRInput.RawButton.LHandTrigger))  move.z -= 1.0f;
        if (move.sqrMagnitude > 0f)
        {
            t.position += move * translationSpeed * Time.deltaTime;
            _hasUnsavedNudges = true;
        }

        // ----- Rotation (A/B pitch, X/Y yaw, R3 roll) -----
        float rotStep = rotationSpeed * Time.deltaTime;
        if (OVRInput.Get(OVRInput.RawButton.A))
            t.rotation = Quaternion.AngleAxis( rotStep, Vector3.right) * t.rotation;
        if (OVRInput.Get(OVRInput.RawButton.B))
            t.rotation = Quaternion.AngleAxis(-rotStep, Vector3.right) * t.rotation;
        if (OVRInput.Get(OVRInput.RawButton.X))
            t.rotation = Quaternion.AngleAxis(-rotStep, Vector3.up) * t.rotation;
        if (OVRInput.Get(OVRInput.RawButton.Y))
            t.rotation = Quaternion.AngleAxis( rotStep, Vector3.up) * t.rotation;
        if (OVRInput.Get(OVRInput.RawButton.RThumbstick))
            t.rotation = Quaternion.AngleAxis( rotStep, Vector3.forward) * t.rotation;

        if (OVRInput.Get(OVRInput.RawButton.A) || OVRInput.Get(OVRInput.RawButton.B)
            || OVRInput.Get(OVRInput.RawButton.X) || OVRInput.Get(OVRInput.RawButton.Y)
            || OVRInput.Get(OVRInput.RawButton.RThumbstick))
        {
            _hasUnsavedNudges = true;
        }
    }
#endif

#if !UNITY_EDITOR
    /// <summary>Right-controller ray in WORLD space. OVRInput gives a TrackingSpace-local
    /// pose, so it must be transformed through the TrackingSpace transform to hit the
    /// world-fixed bar colliders — identical to MultiSessionGridManager.TryGetControllerRay.</summary>
    private bool TryGetControllerRay(out Vector3 origin, out Vector3 direction)
    {
        if (_trackingSpace == null)
        {
            var go = GameObject.Find(trackingSpaceName);
            _trackingSpace = go != null ? go.transform : null;
        }

        Vector3    localPos = OVRInput.GetLocalControllerPosition(OVRInput.Controller.RTouch);
        Quaternion localRot = OVRInput.GetLocalControllerRotation(OVRInput.Controller.RTouch);

        if (_trackingSpace != null)
        {
            origin    = _trackingSpace.TransformPoint(localPos);
            direction = _trackingSpace.rotation * (localRot * Vector3.forward);
        }
        else
        {
            origin    = localPos;
            direction = localRot * Vector3.forward;
        }
        return direction.sqrMagnitude > 0.01f;
    }
#endif

    /// <summary>Scale a bar up and push it toward the viewer when hovered (selector-style),
    /// restoring its resting transform otherwise.</summary>
    private void SetBarHovered(int idx, bool hovered)
    {
        if (_bars[idx].Root == null) return;
        var tr = _bars[idx].Root.transform;
        if (hovered)
        {
            tr.localScale    = _bars[idx].BaseLocalScale * hoverScaleMultiplier;
            // Chart faces +Z toward the viewer, so -Z on the fill quads is "toward viewer".
            tr.localPosition = _bars[idx].BaseLocalPos + new Vector3(0f, 0f, -hoverForwardPush);
        }
        else
        {
            tr.localScale    = _bars[idx].BaseLocalScale;
            tr.localPosition = _bars[idx].BaseLocalPos;
        }
    }

    void OnDisable()
    {
        StopNetMq();
        AdjustModeActive = false;
    }
    void OnDestroy()
    {
        StopNetMq();
        if (_chartRoot != null) Destroy(_chartRoot);  // top-level, not auto-destroyed with this GO
        HoveredBarIndex = -1;
        AdjustModeActive = false;
    }

    // -----------------------------------------------------------------------
    // Chart construction

    private void BuildChart()
    {
        _bars    = new BarEntry[_nSessions];
        _riskRaw = new int[_nSessions];

        _chartRoot = new GameObject("RiskChart");
        _chartRoot.transform.SetParent(transform, false);

        // Title label at the top-right corner of the chart.
        var titleGo = new GameObject("ChartTitle");
        titleGo.transform.SetParent(_chartRoot.transform, false);
        float chartTopY = (_nSessions * (barHeight + barGap)) * 0.5f + titleOffsetY;
        titleGo.transform.localPosition = new Vector3(maxBarWidth * 0.5f, chartTopY, 0f);
        var titleText = CreateTextMesh(titleGo, "RISK", 0.014f, TextAnchor.MiddleRight);
        titleText.color = new Color(0.9f, 0.9f, 0.9f, 1f);

        for (int i = 0; i < _nSessions; i++)
        {
            float yCenter = ((_nSessions - 1 - i) - (_nSessions - 1) * 0.5f)
                            * (barHeight + barGap);
            _bars[i] = BuildBar(i, yCenter);
        }
    }

    private BarEntry BuildBar(int idx, float yCenter)
    {
        var entry  = new BarEntry();
        entry.SessionIndex = idx;
        entry.TopicPort    = _baseTopicPort + idx * _portStep;

        // Root at (0, yCenter, 0) in chart-root space.
        entry.Root = new GameObject($"Bar_{idx:D2}");
        entry.Root.transform.SetParent(_chartRoot.transform, false);
        entry.Root.transform.localPosition = new Vector3(0f, yCenter, 0f);

        // Session label — left of the track.
        var labelGo = new GameObject("Label");
        labelGo.transform.SetParent(entry.Root.transform, false);
        labelGo.transform.localPosition = new Vector3(labelOffsetX, 0f, 0.001f);
        var tm = CreateTextMesh(labelGo, $"S{idx:D2}", fontSize, TextAnchor.MiddleRight);
        bool isCurrent = (idx == SessionRegistry.SelectedSessionIndex);
        // Current-scene bar reads CYAN (label + track); others stay neutral gray.
        tm.color = isCurrent ? new Color(0.6f, 0.95f, 1f, 1f) : new Color(0.75f, 0.75f, 0.75f, 1f);

        // Track background (full width; cyan for the current scene, dark gray otherwise).
        var trackGo = CreateQuad("BarTrack", entry.Root.transform,
            new Vector3(maxBarWidth * 0.5f, 0f, 0f),
            new Vector3(maxBarWidth, barHeight, 0.001f));
        entry.TrackRenderer = trackGo.GetComponent<Renderer>();
        SetQuadColor(entry.TrackRenderer, isCurrent
            ? new Color(0f, 0.55f, 0.65f, 1f)
            : new Color(0.12f, 0.12f, 0.14f, 1f));

        // Fill bar (starts at left edge, scaled on X by raw risk in [0, 1]).
        var fillGo = CreateQuad("BarFill", entry.Root.transform,
            new Vector3(0f, 0f, -0.001f),    // pivot at left edge
            new Vector3(maxBarWidth, barHeight * 0.85f, 0.001f));
        // Pivot at left edge: shift so the fill grows rightward.
        fillGo.transform.localPosition = new Vector3(maxBarWidth * 0.5f, 0f, -0.001f);
        entry.FillTransform = fillGo.transform;
        entry.FillRenderer  = fillGo.GetComponent<Renderer>();
        SetQuadColor(entry.FillRenderer, RiskColor(0f));

        // BoxCollider on the track for ray detection (world-space bounds).
        // Track localScale is (maxBarWidth, barHeight, 0.001). Use normalized local-space
        // size (1,1,1) so world size = scale, but multiply Z by 10 so the box has 0.01 m
        // world depth — enough for bounds.IntersectRay to reliably hit a nearly-flat quad.
        entry.Collider = trackGo.AddComponent<BoxCollider>();
        entry.Collider.size   = new Vector3(1f, 1f, 10f);
        entry.Collider.center = Vector3.zero;

        // Resting transform of the bar Root — restored when un-hovered.
        entry.BaseLocalPos   = entry.Root.transform.localPosition;
        entry.BaseLocalScale = entry.Root.transform.localScale;

        return entry;
    }

    private void UpdateBar(int idx, float risk)
    {
        if (_bars[idx].FillTransform == null) return;
        float t = Mathf.Clamp01(risk);
        var fill = _bars[idx].FillTransform;
        // Preserve Y and Z scale — only X grows with risk. Overwriting with (t,1,1) was
        // wrong: it lost the barHeight*0.85 and 0.001 Z values, making the fill 1 m tall.
        fill.localScale    = new Vector3(t * maxBarWidth, barHeight * 0.85f, 0.001f);
        // Re-anchor to left edge: at t=1 center = maxBarWidth/2; at t=0 center = 0.
        fill.localPosition = new Vector3(maxBarWidth * 0.5f * t, 0f, -0.001f);
        SetQuadColor(_bars[idx].FillRenderer, RiskColor(risk));
    }

    private void SetTrackHighlight(int idx, bool hovered)
    {
        if (_bars[idx].TrackRenderer == null) return;
        bool isCurrent = (idx == SessionRegistry.SelectedSessionIndex);
        Color c = isCurrent ? new Color(0f, 0.55f, 0.65f, 1f)     // cyan = the scene we are in
                            : new Color(0.12f, 0.12f, 0.14f, 1f);
        if (hovered) c = new Color(c.r + 0.15f, c.g + 0.15f, c.b + 0.15f, 1f);
        SetQuadColor(_bars[idx].TrackRenderer, c);
    }

    // -----------------------------------------------------------------------
    // NetMQ

    private void OpenSubscribers()
    {
        _subs    = new SubscriberSocket[_nSessions];
        _pollers = new NetMQPoller[_nSessions];
        for (int i = 0; i < _nSessions; i++)
            OpenOne(i);
    }

    private void OpenOne(int idx)
    {
        int port = _baseTopicPort + idx * _portStep;
        try
        {
            var sub = new SubscriberSocket();
            sub.Options.ReceiveHighWatermark = 2;
            sub.Options.Linger = TimeSpan.Zero;
            sub.Connect($"tcp://{_ip}:{port}");
            sub.Subscribe("SimPub/Status/risk");
            int captured = idx;  // capture for lambda
            sub.ReceiveReady += (s, e) => OnMsg(e, captured);
            var poller = new NetMQPoller { sub };
            poller.RunAsync();
            _subs[idx]    = sub;
            _pollers[idx] = poller;
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"[RiskBarChart] Failed to open subscriber for session {idx} "
                           + $"tcp://{_ip}:{port}: {ex.Message}");
        }
    }

    private void OnMsg(NetMQSocketEventArgs e, int idx)
    {
        if (_shuttingDown) return;
        try
        {
            var msg = new NetMQMessage();
            bool got;
            try { got = e.Socket.TryReceiveMultipartMessage(ref msg); }
            catch (TerminatingException) { _shuttingDown = true; return; }
            catch (ObjectDisposedException) { _shuttingDown = true; return; }
            if (!got || msg.FrameCount < 2) return;

            if (float.TryParse(msg[1].ConvertToString(),
                System.Globalization.NumberStyles.Float,
                System.Globalization.CultureInfo.InvariantCulture, out float r))
            {
                Interlocked.Exchange(ref _riskRaw[idx],
                    System.BitConverter.ToInt32(System.BitConverter.GetBytes(r), 0));
            }
        }
        catch { }
    }

    private void ShutdownOne(int idx)
    {
        var sub    = _subs?[idx];
        var poller = _pollers?[idx];
        if (_subs    != null) _subs[idx]    = null;
        if (_pollers != null) _pollers[idx] = null;

        try { if (sub != null) sub.ReceiveReady -= (s, e) => OnMsg(e, idx); } catch { }
        try { if (poller != null && poller.IsRunning) poller.Stop(); } catch { }
        try { sub?.Close(); }    catch { }
        try { sub?.Dispose(); }  catch { }
        try { poller?.Dispose(); } catch { }
    }

    // -----------------------------------------------------------------------
    // Helpers

    private static Color RiskColor(float risk)
    {
        float t = Mathf.Clamp01(risk);
        return t < 0.5f
            ? Color.Lerp(Color.green,  Color.yellow, t * 2f)
            : Color.Lerp(Color.yellow, Color.red,    (t - 0.5f) * 2f);
    }

    private void SavePose(Vector3 pos, Quaternion rot)
    {
        PlayerPrefs.SetString(playerPrefsKeyPos, $"{pos.x},{pos.y},{pos.z}");
        var e = rot.eulerAngles;
        PlayerPrefs.SetString(playerPrefsKeyRot, $"{e.x},{e.y},{e.z}");
        PlayerPrefs.Save();
        Debug.Log($"[RiskBarChart] saved pose to PlayerPrefs: pos={pos} euler={e}");
    }

    private bool TryLoadPose(out Vector3 pos, out Quaternion rot)
    {
        pos = Vector3.zero;
        rot = Quaternion.identity;
        if (!PlayerPrefs.HasKey(playerPrefsKeyPos) || !PlayerPrefs.HasKey(playerPrefsKeyRot))
            return false;
        try
        {
            var pStr = PlayerPrefs.GetString(playerPrefsKeyPos).Split(',');
            var rStr = PlayerPrefs.GetString(playerPrefsKeyRot).Split(',');
            if (pStr.Length != 3 || rStr.Length != 3) return false;
            pos = new Vector3(float.Parse(pStr[0]), float.Parse(pStr[1]), float.Parse(pStr[2]));
            rot = Quaternion.Euler(float.Parse(rStr[0]), float.Parse(rStr[1]), float.Parse(rStr[2]));
            return true;
        }
        catch (System.Exception ex)
        {
            Debug.LogWarning($"[RiskBarChart] PlayerPrefs pose parse failed: {ex.Message}");
            return false;
        }
    }

    private static GameObject CreateQuad(string name, Transform parent,
        Vector3 localPos, Vector3 localScale)
    {
        var go = GameObject.CreatePrimitive(PrimitiveType.Quad);
        go.name = name;
        go.transform.SetParent(parent, false);
        go.transform.localPosition = localPos;
        go.transform.localScale    = localScale;
        // Destroy the MeshCollider that CreatePrimitive adds.
        var col = go.GetComponent<Collider>();
        if (col != null) Destroy(col);
        // Assign the project-local UnlitDoubleSided shader asset so URP variant stripping
        // on Android doesn't render everything magenta. CreatePrimitive keeps the default
        // URP/Lit material whose variants get stripped from Android builds.
        var mr = go.GetComponent<MeshRenderer>();
        if (mr != null)
        {
            Shader sh = Shader.Find("Custom/UnlitDoubleSided") ?? Shader.Find("Unlit/Color");
            if (sh != null) mr.material = new Material(sh);
            mr.shadowCastingMode    = ShadowCastingMode.Off;
            mr.receiveShadows       = false;
            mr.lightProbeUsage      = LightProbeUsage.Off;
            mr.reflectionProbeUsage = ReflectionProbeUsage.Off;
        }
        return go;
    }

    private static void SetQuadColor(Renderer r, Color c)
    {
        if (r == null) return;
        var mpb = new MaterialPropertyBlock();
        r.GetPropertyBlock(mpb);
        mpb.SetColor("_Color",     c);
        mpb.SetColor("_BaseColor", c);
        r.SetPropertyBlock(mpb);
    }

    private static TextMesh CreateTextMesh(GameObject go, string text,
        float charSize, TextAnchor anchor)
    {
        var tm = go.AddComponent<TextMesh>();
        var font = Resources.GetBuiltinResource<Font>("Arial.ttf");
        if (font != null) tm.font = font;
        tm.text          = text;
        tm.fontSize      = 46;
        tm.characterSize = charSize;
        tm.anchor        = anchor;
        tm.alignment     = TextAlignment.Center;
        tm.color         = new Color(0.85f, 0.85f, 0.85f, 1f);
        var mr = go.GetComponent<MeshRenderer>();
        if (mr != null)
        {
            if (font != null) mr.sharedMaterial = font.material;
            mr.shadowCastingMode    = ShadowCastingMode.Off;
            mr.receiveShadows       = false;
            mr.lightProbeUsage      = LightProbeUsage.Off;
            mr.reflectionProbeUsage = ReflectionProbeUsage.Off;
        }
        return tm;
    }
}
