using System;                       // StringComparison (OOD owner check)
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

/// <summary>
/// Builds a 5x3 grid of session-thumbnail panels in the selector scene.
/// Handles right-controller ray-cast + trigger interaction to select a session,
/// then loads SIIScene_InterventionV1.
/// </summary>
public class MultiSessionGridManager : MonoBehaviour
{
    // -----------------------------------------------------------------------
    // Inspector

    [Header("Session Config")]
    [Tooltip("15 labels, one per session (e.g. trajectory filename or short description).")]
    public string[] sessionLabels = new string[15];
    [Tooltip("Fallback publisher IP. Used only if auto-discovery is off or no beacon "
           + "arrives before discoveryTimeoutSeconds. With discovery on, the launcher's "
           + "--host overrides this — so the IP is no longer hardcoded in the scene.")]
    public string publisherIp     = "127.0.0.1";
    public int    baseTopicPort   = 7741;
    public int    portStep        = 10;
    public string thumbnailTopic  = "SimPub/Sensors/front/rgb";

    [Header("Auto-Discovery")]
    [Tooltip("Listen for the launcher's UDP beacon and use the IP/ports it advertises "
           + "instead of the serialized publisherIp. The IP then lives only in the "
           + "launcher's --host. Turn off to always use the serialized publisherIp.")]
    public bool  useDiscovery = true;
    [Tooltip("UDP port to listen on for the discovery beacon. Must match the launcher's "
           + "--discovery_port (default 8720; not SimPub's 7720 multicast port).")]
    public int   discoveryPort = 8720;
    [Tooltip("Max seconds to wait for a discovery beacon before spawning the grid with "
           + "the fallback publisherIp. The grid still respawns automatically if a beacon "
           + "arrives later (e.g. launcher started after the headset).")]
    public float discoveryTimeoutSeconds = 8.0f;
    [Tooltip("Also adopt baseTopicPort/portStep from the beacon, not just the IP.")]
    public bool  overridePortsFromDiscovery = true;

    [Header("Grid Layout")]
    public int   columns       = 5;
    public int   rows          = 3;
    [Tooltip("Fill panels column-by-column (session 0 = top-left, then DOWN 3, then the "
           + "next column). Each group of 3 sessions = one full column, so a variable "
           + "session count (3/6/9/12/15) fills whole columns and the empty panels hide "
           + "themselves. Off = legacy row-major fill.")]
    public bool  columnMajorFill = true;
    public float panelWidth    = 0.38f;
    public float panelHeight   = 0.27f;
    public float panelGap      = 0.04f;
    public float gridDistance  = 1.8f;   // sphere radius — distance from head to each panel
    public float gridHeightOffset = 0.0f; // vertical offset from eye level (unused in spherical mode)
    [Tooltip("Horizontal angle between adjacent column centres (degrees). "
           + "Default 13° matches the previous 1.8 m flat layout spread.")]
    public float horizontalAngleDeg = 13.0f;
    [Tooltip("Vertical angle between adjacent row centres (degrees). "
           + "Default 10° matches the previous 1.8 m flat layout spread.")]
    public float verticalAngleDeg   = 10.0f;

    [Header("Panel Appearance")]
    [Tooltip("Optional prefab for each panel. If null a minimal quad + border is created at runtime.")]
    public GameObject panelPrefab;
    public Material   panelMaterial;    // unlit, for the thumbnail quad
    public Material   borderMaterial;   // unlit, for the border quad

    [Header("Ray Interaction")]
    public float rayMaxDistance   = 4.0f;
    public int   panelLayerMask   = -1;  // set to the SessionPanel layer in inspector
    [Tooltip("Name of the TrackingSpace transform (parent of controller anchors).")]
    public string trackingSpaceName = "TrackingSpace";
    [Range(0f, 0.95f)]
    [Tooltip("Low-pass smoothing on the controller ray direction. 0 = raw (twitchy), higher = "
           + "steadier but laggier. Makes panels easier to hold/target. 0.5 is a moderate default.")]
    public float raySmoothing = 0.5f;
    [Tooltip("Radius of the SphereCast used for panel hover/select. A thicker pointer makes "
           + "grazing hits count so panels are easier to target, without enlarging the colliders "
           + "themselves (which are tightly spaced and would overlap). 0 falls back to a thin raycast.")]
    public float hoverSphereCastRadius = 0.025f;
    [Tooltip("Hold B this long to broadcast pause/resume to ALL live windows. A short tap "
           + "affects only the hovered window; with nothing hovered, B is left to the "
           + "single-view forwarder.")]
    public float resetAllHoldSeconds = 0.7f;
    [Tooltip("Draw the pointer as a visible beam (ControllerRayVisual, auto-added). The ray "
           + "itself is unchanged — this only renders the origin/direction/hit already computed "
           + "below, so the beam and the hover test can never disagree. In combined view the "
           + "grid stays loaded during single view, so the beam is available for aiming at the "
           + "grid from inside a live session too.")]
    public bool showRayVisual = true;
    [Tooltip("Seconds Left Menu must be held CONTINUOUSLY to toggle the selector's grid-adjust "
           + "mode. Mirrors SceneAnchorManager.adjustToggleHoldSeconds — same button, so keep "
           + "them equal. A tap does nothing. 0 restores the old instant toggle.")]
    public float adjustToggleHoldSeconds = AdjustModeHoldGate.DefaultHoldSeconds;

    [Header("OOD Active Cell Limit")]
    [Tooltip("When enabled, panels whose ACC/risk is >= accOodThreshold are OOD. "
           + "Only the top maxActiveOodCells OOD panels stay active; lower-priority "
           + "OOD panels are explicitly paused and labeled PAUSED (OOD).")]
    public bool enableOodActiveCellLimit = true;
    [Tooltip("ACC/risk threshold at or above which a session counts as OOD.")]
    // OOD remains gated by the launcher's explicit ood_enabled beacon. A positive
    // default makes Quest-owned OOD usable without a scene-specific inspector override,
    // while the launcher flag still keeps it completely off during the recovery baseline.
    public float accOodThreshold = 0.60f;
    [Tooltip("Maximum OOD cells allowed to remain active/running at once. Default 2.")]
    public int maxActiveOodCells = 2;
    [Tooltip("Seconds between OOD limit evaluations.")]
    public float oodEvaluateIntervalSeconds = 0.5f;

    [Header("Standalone Single Session")]
    [Tooltip("When the discovery beacon advertises exactly ONE session, skip the grid "
           + "entirely and auto-select session 0 (standalone SINGLE=1 mode). Reuses the "
           + "normal select → ENTER_SINGLE → load path. Turn off to always show the grid.")]
    public bool  autoEnterWhenSingleSession = true;
    [Tooltip("Settle delay before auto-entering the single session — lets the panel spawn "
           + "and the command socket warm so ENTER_SINGLE lands reliably.")]
    public float autoEnterDelaySeconds = 0.4f;

    [Header("Scene Transition")]
    public string interventionSceneName = "SIIScene_InterventionV1";
    [Tooltip("Seconds to let ENTER_SINGLE leave the PUSH socket before local NetMQ teardown.")]
    public float transitionCommandFlushSeconds = 0.30f;
    [Tooltip("Fallback only. Normal scene switches use local socket cleanup to avoid poisoning new subscribers.")]
    public bool forceGlobalNetMqCleanupOnSceneSwitch = false;

    [Header("No-signal Fallback")]
    public Color noSignalColor = new Color(0.45f, 0.45f, 0.50f, 1f); // visible grey

    [Header("Combined View")]
    [Tooltip("Keep the grid loaded (additive) and selectable while single view is active, "
           + "instead of the old exclusive scene swap. Turn OFF to fall back to today's "
           + "exact behavior if combined view causes performance problems on-device.")]
    [SerializeField] private bool enableCombinedView = true;

    /// <summary>Static mirror of enableCombinedView, set at Start() before the selector scene
    /// can ever hand off to SIIScene_InterventionV1 — readable from InterventionSessionBootstrap
    /// and InterventionBackButton in the other scene without needing a cross-scene reference.</summary>
    public static bool CombinedViewEnabled;

    /// <summary>True when the grid ray is currently hovering a selectable (revealed) panel.
    /// InterventionBackButton reads this to suppress its own exit-trigger fire on the same
    /// RIGHT index trigger button, so aiming at a panel always wins over exiting to pure grid.</summary>
    public static bool IsHoveringSelectablePanel { get; private set; }

    /// <summary>The pointer ray as actually used for hover/select, republished each frame for
    /// ControllerRayVisual to draw. Exposed rather than recomputed so the drawn beam matches the
    /// hit test exactly: the direction here is already low-passed (raySmoothing) and the hit
    /// comes from the thick SphereCast (hoverSphereCastRadius), neither of which a raw
    /// controller-pose laser would reproduce.
    ///
    /// RayFrame is the Time.frameCount of the last publish; a stale value means the grid is not
    /// pointing right now (transitioning, no controller pose, or grid unloaded) and the beam
    /// hides itself. Static so the visual works from either scene under combined view.</summary>
    public static Vector3 RayOrigin    { get; private set; }
    public static Vector3 RayDirection { get; private set; }
    public static bool    RayHasHit    { get; private set; }
    public static Vector3 RayHitPoint  { get; private set; }
    public static int     RayFrame     { get; private set; } = -10;

    // -----------------------------------------------------------------------
    // Private state

    private readonly List<SessionThumbnailPanel> _panels = new();
    private SessionThumbnailPanel _hoveredPanel;
    private Transform _trackingSpace;
    // Read only in the device build (ray smoothing lives under #if !UNITY_EDITOR), so the editor
    // compile sees an assigned-but-never-read field. Silence the editor-only CS0414.
#pragma warning disable 0414
    private Vector3 _smoothedRayDir = Vector3.zero;  // low-pass state for raySmoothing (zero = uninitialized)
#pragma warning restore 0414
    private bool _gridSpawned;
    private bool _transitioning;
    private PublisherDiscoveryListener _discovery;
    private string _activeIp;  // IP the currently-spawned panels are using
    private string _pendingEnterIp;   // session selected this frame; ENTER_SINGLE resent in ShutdownAndLoad
    private int    _pendingEnterPort;
    private int _activeBaseTopicPort;
    private int _activePortStep;
    private int _activeSessionCount;
    private int _activePanelCount;
    private bool _bPressPending;
    private bool _bBroadcastSent;
    private float _bPressStartTime;
    private SessionThumbnailPanel _bPressTarget;
    private readonly HashSet<int> _oodPausedSessions = new();
    private readonly Dictionary<int, float> _lastOodPauseCommandAt = new();
    private float _nextOodEvaluateAt;

    // -----------------------------------------------------------------------

    [Header("Spawn Delay")]
    [Tooltip("Seconds to wait after scene load before spawning the grid on COLD start. "
           + "Allows VR tracking to stabilize so the grid spawns in front of the user.")]
    public float spawnDelaySeconds = 2.5f;
    [Tooltip("Seconds to wait before spawning the grid when RETURNING from single view "
           + "(warm re-entry). Tracking is already stable, so this is tiny — it removes the "
           + "~2.5s cold-start delay from the exit-to-grid path. Only a one-frame settle margin.")]
    public float warmReentrySpawnDelaySeconds = 0.15f;

    // Statics survive scene reloads: true once the grid has spawned at least once this
    // app session. Used to distinguish a cold app start (full spawnDelaySeconds, needed for
    // VR tracking to stabilize) from a warm return out of single view (tiny delay).
    private static bool s_hasSpawnedOnce;

    void Start()
    {
        CombinedViewEnabled = enableCombinedView;

        // Always clear any stale session selection when entering the selector.
        // Prevents InterventionSessionBootstrap from auto-reconfiguring on next launch.
        // Safe under combined view too: this Start() only ever runs once (the grid scene
        // is never unloaded in combined mode), so it only fires before any selection exists.
        SessionRegistry.Clear();
        _trackingSpace = FindTrackingSpace();

        // Start UDP auto-discovery so the publisher IP comes from the launcher's
        // --host, not a value baked into the scene. Auto-added so no Editor wiring
        // is needed; the serialized publisherIp remains the fallback.
        if (useDiscovery)
        {
            _discovery = GetComponent<PublisherDiscoveryListener>()
                      ?? gameObject.AddComponent<PublisherDiscoveryListener>();
            _discovery.discoveryPort = discoveryPort;
        }

        // Visible pointer beam. Auto-added like the discovery listener so no Editor wiring is
        // needed and it lives with the grid: it therefore renders in the pure selector AND,
        // under combined view, while single view is loaded on top — which is exactly when the
        // operator needs to see where they are aiming to switch sessions.
        if (showRayVisual)
        {
            var rayVisual = GetComponent<ControllerRayVisual>()
                         ?? gameObject.AddComponent<ControllerRayVisual>();
            rayVisual.fallbackLength = rayMaxDistance;
        }

        // Delay spawn so Quest tracking is stable, then give discovery a moment to
        // hear a beacon before committing to an IP. Cold start waits the full delay;
        // a warm return from single view uses the tiny delay (tracking already stable).
        bool warm = s_hasSpawnedOnce;
        float effectiveDelay = warm ? warmReentrySpawnDelaySeconds : spawnDelaySeconds;
        StartCoroutine(SpawnGridWhenReady(effectiveDelay));
        Debug.Log($"[MultiSessionGridManager] Start — grid spawns in ~{effectiveDelay}s "
                + $"({(warm ? "warm re-entry" : "cold start")}). "
                + $"useDiscovery={useDiscovery} discoveryPort={discoveryPort} fallbackIp={publisherIp}");
    }

    /// <summary>Wait for VR tracking to stabilize, then (if discovery is on) wait up to
    /// discoveryTimeoutSeconds for a beacon before spawning. Falls back to the serialized
    /// publisherIp if no beacon arrives — the grid still respawns later via Update if one does.</summary>
    private System.Collections.IEnumerator SpawnGridWhenReady(float spawnDelay)
    {
        yield return new WaitForSeconds(Mathf.Max(0f, spawnDelay));

        if (useDiscovery && _discovery != null)
        {
            float deadline = Time.unscaledTime + Mathf.Max(0f, discoveryTimeoutSeconds);
            // On warm re-entry the listener seeds DiscoveredIp from the static beacon cache
            // in OnEnable, so HasDiscovered is already true and this loop falls straight
            // through — no waiting for the next ~1/sec beacon.
            while (!_discovery.HasDiscovered && Time.unscaledTime < deadline)
                yield return null;
            if (!_discovery.HasDiscovered)
                Debug.LogWarning($"[MultiSessionGridManager] No discovery beacon within "
                    + $"{discoveryTimeoutSeconds}s — using fallback IP {publisherIp}. "
                    + "Will respawn automatically if a beacon arrives later.");
        }

        SpawnGrid();
    }

    void Update()
    {
        if (_transitioning) return;
        UpdateRay();
        HandleSelectorAdjustToggle();
        HandleReposition();
        HandleDiscoveryIpChange();
        EnforceOodActiveCellLimit();
    }

    /// <summary>
    /// True while the grid may be repositioned. The grid is now MODAL, exactly like the point
    /// cloud / RGB diamond anchor: you must be in adjust mode first, so a stray RIGHT grip can
    /// no longer teleport the whole grid mid-session.
    ///
    /// Previously this was the INVERSE — reposition was blocked *during* PC-adjust mode to stop
    /// it fighting the anchor nudges, and free at all other times.
    ///
    /// Which toggle owns adjust mode depends on what is loaded, because SceneAnchorManager only
    /// exists in SIIScene_InterventionV1:
    ///   * single view / combined view -> SceneAnchorManager.AdjustModeActive, so ONE Left Menu
    ///     press puts the anchor and the grid into adjust mode together (they use disjoint
    ///     controls: the anchor takes sticks/L-triggers/A/B/X/Y, the grid takes RIGHT grip).
    ///   * pure selector -> the local toggle below, on the SAME Left Menu button. Without it the
    ///     grid would be permanently unmovable whenever the single-view scene is unloaded,
    ///     which is the state the operator is in at cold start.
    /// </summary>
    public static bool GridAdjustModeActive =>
        SceneAnchorManager.Exists ? SceneAnchorManager.AdjustModeActive : s_selectorAdjustMode;

    private static bool s_selectorAdjustMode;
    private readonly AdjustModeHoldGate _menuHold = new AdjustModeHoldGate();

    /// <summary>Left Menu toggle for the pure selector, mirroring SceneAnchorManager's. Runs ONLY
    /// when no SceneAnchorManager is alive, so the two can never both consume the same press.
    /// Reset whenever one appears, so entering a session always starts locked (the anchor's own
    /// default) rather than inheriting a stale selector unlock — and so a hold started here can
    /// never complete after the anchor has taken the button over.
    ///
    /// Uses the SAME AdjustModeHoldGate as SceneAnchorManager: one physical button, one
    /// definition of what counts as a hold.</summary>
    private void HandleSelectorAdjustToggle()
    {
        if (SceneAnchorManager.Exists)
        {
            s_selectorAdjustMode = false;
            _menuHold.Reset();
            return;
        }

        // Deliberately NOT wrapped in #if !UNITY_EDITOR, matching HandleReposition below (the
        // only other OVRInput reader in this class that is unguarded). In the editor OVRInput
        // simply reports false, so this is inert there.
        bool pressed = OVRInput.Get(OVRInput.RawButton.Start) && !RiskBarChart.AdjustModeActive;
        if (_menuHold.Poll(pressed, adjustToggleHoldSeconds))
        {
            s_selectorAdjustMode = !s_selectorAdjustMode;
            Debug.Log($"[MultiSessionGridManager] Grid adjust mode = {s_selectorAdjustMode} "
                    + $"(Left Menu held {adjustToggleHoldSeconds:F1}s; RIGHT grip repositions "
                    + "the grid while unlocked).");
        }
    }

    /// <summary>RIGHT grip (RHandTrigger) repositions the grid in front of the current head
    /// direction — now only while in adjust mode (see GridAdjustModeActive). Moved off Y so Y is
    /// free for the single-view sim start/stop toggle in combined view (both scripts run at once
    /// there). Still suppressed during an active intervention, where RIGHT grip is the MC
    /// orientation clutch.</summary>
    private void HandleReposition()
    {
        if (InterventionStatusHud.InterventionActive) return;

        // Belt and braces with the line above: under MC the whole right controller is the
        // operator's hand on the arm, and the lock outlives any single frame of HUD state.
        if (MotionControllerModeManager.McControlsLocked) return;

        if (!GridAdjustModeActive) return;

        if (OVRInput.GetDown(OVRInput.RawButton.RHandTrigger))
        {
            Debug.Log("[MultiSessionGridManager] RIGHT grip in adjust mode — repositioning grid.");
            RespawnGrid();
        }
    }

    /// <summary>If a discovery beacon advertises a different session layout than
    /// the live panels are using, rebuild the grid so it follows the current feed.</summary>
    private void HandleDiscoveryIpChange()
    {
        if (!useDiscovery || _discovery == null || !_gridSpawned) return;
        if (!_discovery.HasDiscovered) return;

        int nextBaseTopicPort = _discovery.DiscoveredBaseTopicPort > 0
                              ? _discovery.DiscoveredBaseTopicPort
                              : baseTopicPort;
        int nextPortStep = _discovery.DiscoveredPortStep > 0
                         ? _discovery.DiscoveredPortStep
                         : portStep;
        int nextSessionCount = ResolveSessionCount();
        int nextPanelCount = ResolvePanelCount();

        if (_discovery.DiscoveredIp == _activeIp
            && nextBaseTopicPort == _activeBaseTopicPort
            && nextPortStep == _activePortStep
            && nextSessionCount == _activeSessionCount
            && nextPanelCount == _activePanelCount)
            return;

        Debug.Log($"[MultiSessionGridManager] Discovery layout changed "
                + $"ip {_activeIp}→{_discovery.DiscoveredIp}, "
                + $"base {_activeBaseTopicPort}→{nextBaseTopicPort}, "
                + $"step {_activePortStep}→{nextPortStep}, "
                + $"sessions {_activeSessionCount}→{nextSessionCount}, "
                + $"panels {_activePanelCount}→{nextPanelCount} — respawning grid.");
        RespawnGrid();
    }

    private bool HasAdvertisedSessionCount()
    {
        return useDiscovery
            && _discovery != null
            && _discovery.HasDiscovered
            && _discovery.DiscoveredSessions > 0;
    }

    private int ResolveSessionCount()
    {
        int maxCount = Mathf.Max(1, columns * rows);
        if (HasAdvertisedSessionCount())
            return Mathf.Clamp(_discovery.DiscoveredSessions, 1, maxCount);
        return maxCount;
    }

    /// <summary>Number of visible grid slots. Fleet studies advertise a fixed capacity
    /// (normally nine) separately from the number of active publisher sessions. The
    /// extra panels are inert placeholders: they never open sockets or accept input.</summary>
    private int ResolvePanelCount()
    {
        int activeCount = ResolveSessionCount();
        int maxCount = Mathf.Max(1, columns * rows);
        if (HasAdvertisedSessionCount() && _discovery.DiscoveredGridCapacity > 0)
            return Mathf.Clamp(Mathf.Max(activeCount, _discovery.DiscoveredGridCapacity), 1, maxCount);
        return activeCount;
    }

    /// <summary>Destroy the current panels and rebuild from scratch (used by Y-reposition
    /// and discovery IP changes).</summary>
    private void RespawnGrid()
    {
        foreach (var p in _panels)
            if (p != null) Destroy(p.gameObject);
        _panels.Clear();
        _hoveredPanel = null;
        // Clear ALL OOD bookkeeping, not just two of it. _oodCommandedPaused and
        // _oodManualOverrideUntil used to survive a respawn, so after a grid rebuild the
        // supervisor could believe it had parked a session it never touched (and resume a
        // session the operator had paused), or stay locked in a stale override window.
        _oodPausedSessions.Clear();
        _lastOodPauseCommandAt.Clear();
        _oodCommandedPaused.Clear();
        _oodManualOverrideUntil.Clear();
        _oodSkipLogged.Clear();
        _gridSpawned = false;
        SpawnGrid();
    }

    // -----------------------------------------------------------------------
    // Grid construction

    private void SpawnGrid()
    {
        if (_gridSpawned) return;
        _gridSpawned = true;
        // Mark that the grid has spawned at least once this app session so the next
        // selector entry (warm re-entry from single view) uses the tiny spawn delay.
        s_hasSpawnedOnce = true;

        // Adopt the discovered IP (and optionally ports) so panels connect to the
        // launcher advertised by the beacon, overriding the serialized fallback.
        // rgbMode is advertised the same way: the launcher started with --rgb_mode
        // means the selected session's single view should render diamond RGB
        // camera panels instead of point clouds (read by InterventionSessionBootstrap).
        bool discoveredRgbMode = false;
        if (useDiscovery && _discovery != null && _discovery.HasDiscovered)
        {
            publisherIp = _discovery.DiscoveredIp;
            discoveredRgbMode = _discovery.DiscoveredRgbMode;
            if (overridePortsFromDiscovery)
            {
                if (_discovery.DiscoveredBaseTopicPort > 0) baseTopicPort = _discovery.DiscoveredBaseTopicPort;
                if (_discovery.DiscoveredPortStep      > 0) portStep      = _discovery.DiscoveredPortStep;
            }
        }

        int activeCount = ResolveSessionCount();
        int panelCount = ResolvePanelCount();
        bool showExpectedPlaceholders = HasAdvertisedSessionCount();
        _activeIp = publisherIp;
        _activeBaseTopicPort = baseTopicPort;
        _activePortStep = portStep;
        _activeSessionCount = activeCount;
        _activePanelCount = panelCount;

        // Resolve head position and horizontal forward direction.
        Camera mainCam = Camera.main;
        Vector3 headPos = mainCam != null ? mainCam.transform.position : Vector3.up * 1.4f;
        Vector3 fwd     = mainCam != null ? mainCam.transform.forward : Vector3.forward;
        fwd.y = 0f;
        if (fwd.sqrMagnitude < 0.001f) fwd = Vector3.forward;
        fwd.Normalize();
        // World-space right perpendicular to the projected forward direction.
        Vector3 camRight = Vector3.Cross(Vector3.up, fwd).normalized;

        for (int i = 0; i < panelCount; i++)
        {
            // Column-major: session 0,1,2 fill the left column top→bottom, 3,4,5 the
            // next column, etc. — so each group of 3 sessions is one full column and a
            // variable session count fills whole columns. Legacy row-major fills the
            // top row first (0,1,2,3,4) instead.
            int col, row;
            if (columnMajorFill) { col = i / rows;    row = i % rows; }
            else                 { col = i % columns; row = i / columns; }

            // Signed offsets from the grid centre (-2…+2 columns, -1…+1 rows).
            float colOffset = col - (columns - 1) * 0.5f;
            float rowOffset = row - (rows    - 1) * 0.5f;

            // Horizontal rotation around world-up, then vertical rotation around
            // the resulting local right vector — places each panel on a sphere
            // surface centred at the headset, regardless of look direction.
            Quaternion hRot   = Quaternion.AngleAxis(colOffset * horizontalAngleDeg, Vector3.up);
            Vector3    hFwd   = hRot * fwd;
            Vector3    hRight = hRot * camRight;
            Quaternion vRot   = Quaternion.AngleAxis(-rowOffset * verticalAngleDeg, hRight);
            Vector3    dir    = (vRot * hFwd).normalized;

            Vector3 worldPos = headPos + dir * gridDistance;

            GameObject panelGo = CreatePanelGameObject(i, worldPos);
            panelGo.transform.position = worldPos;
            // Every panel faces directly toward the headset — guaranteed visibility
            // from any look direction, no panel ever ends up edge-on to the viewer.
            Vector3 toHead = headPos - worldPos;
            if (toHead.sqrMagnitude > 0.01f)
                panelGo.transform.rotation = Quaternion.LookRotation(toHead, Vector3.up);

            SessionThumbnailPanel panel = panelGo.GetComponent<SessionThumbnailPanel>();
            bool inactive = i >= activeCount;
            panel.sessionIndex   = i;
            panel.topicPort      = baseTopicPort + i * portStep;
            panel.commandPort    = panel.topicPort + 5;
            panel.publisherIp    = publisherIp;
            panel.sessionLabel   = inactive
                                    ? $"Robot {i + 1} — INACTIVE"
                                    : (i < sessionLabels.Length && !string.IsNullOrEmpty(sessionLabels[i])
                                        ? sessionLabels[i]
                                        : $"Session {i}");
            panel.thumbnailTopic = thumbnailTopic;
            panel.rgbMode        = discoveredRgbMode;
            panel.inactive       = inactive;
            panel.showPlaceholderUntilFrame = showExpectedPlaceholders;
            panel.placeholderColor = noSignalColor;

            // Keep every command socket warm. Selection broadcasts EXIT_SINGLE to all
            // non-selected runtimes before entering the target; cold PUSH sockets could
            // otherwise drop that lifecycle cleanup during a fast scene switch.
            if (!inactive)
                SessionCommandSender.Warm(panel.publisherIp, panel.commandPort);

            // Update the label text if it was already created
            var labelTxt = panelGo.GetComponentInChildren<UnityEngine.UI.Text>();
            if (labelTxt != null)
                labelTxt.text = panel.sessionLabel;

            _panels.Add(panel);
        }

        Debug.Log($"[MultiSessionGridManager] Spawned {activeCount} active sessions in " +
                  $"{panelCount} panels (spherical) " +
                  $"headPos={headPos} fwd={fwd} publisherIp={publisherIp} " +
                  $"baseTopic={baseTopicPort} step={portStep} placeholders={showExpectedPlaceholders}");

        // Standalone SINGLE=1: the launcher advertised exactly one session, so skip the
        // grid and auto-select session 0 — reusing the normal manual-select path.
        if (autoEnterWhenSingleSession && activeCount == 1 && panelCount == 1 && HasAdvertisedSessionCount()
            && _panels.Count == 1 && _panels[0] != null && !_transitioning)
        {
            Debug.Log("[MultiSessionGridManager] Single session advertised — auto-entering single view.");
            StartCoroutine(AutoEnterSingle(_panels[0]));
        }
    }

    /// <summary>Standalone single-session entry: warm the command socket, settle briefly,
    /// then run the same select → ENTER_SINGLE → ShutdownAndLoad sequence as a manual
    /// trigger-select (UpdateRay), so the headset lands in single view without the grid.</summary>
    private System.Collections.IEnumerator AutoEnterSingle(SessionThumbnailPanel panel)
    {
        SessionCommandSender.Warm(panel.publisherIp, panel.commandPort);
        yield return new WaitForSecondsRealtime(Mathf.Max(0f, autoEnterDelaySeconds));

        if (_transitioning || panel == null) yield break;

        ExitAllOtherSessions(panel, "auto-select");
        panel.Select();
        _pendingEnterIp   = panel.publisherIp;
        _pendingEnterPort = panel.commandPort;
        bool enterSent = SessionCommandSender.Send(_pendingEnterIp, _pendingEnterPort, "ENTER_SINGLE");
        _transitioning = true;
        Debug.Log(
            $"[MultiSessionGridManager] Auto-entered session {panel.sessionIndex} → "
            + $"ENTER_SINGLE result={enterSent} then loading {interventionSceneName}"
        );
        StartCoroutine(ShutdownAndLoad(interventionSceneName));
    }

    private void ExitAllOtherSessions(SessionThumbnailPanel selected, string reason)
    {
        int attempted = 0;
        int sent = 0;
        foreach (SessionThumbnailPanel panel in _panels)
        {
            if (panel == null || panel == selected || panel.IsInactive)
                continue;
            attempted++;
            if (SessionCommandSender.Send(panel.publisherIp, panel.commandPort, "EXIT_SINGLE"))
                sent++;
        }
        Debug.Log($"[MultiSessionGridManager] {reason}: EXIT_SINGLE sent to "
                + $"{sent}/{attempted} non-selected sessions before ENTER_SINGLE.");
    }

    private GameObject CreatePanelGameObject(int index, Vector3 worldPos)
    {
        if (panelPrefab != null)
        {
            var go = Instantiate(panelPrefab, worldPos, Quaternion.identity, transform);
            if (go.GetComponent<SessionThumbnailPanel>() == null)
                go.AddComponent<SessionThumbnailPanel>();
            if (go.GetComponent<BoxCollider>() == null)
                go.AddComponent<BoxCollider>();
            return go;
        }

        // --- Runtime-generated panel (no prefab) ---
        var root = new GameObject($"Panel_{index}");
        root.transform.SetParent(transform, false);

        // Thumbnail quad — must use an unlit material so the texture shows without
        // depending on scene lighting. URP uses "Universal Render Pipeline/Unlit",
        // legacy fallback is "Unlit/Texture".
        var thumbGo = GameObject.CreatePrimitive(PrimitiveType.Quad);
        thumbGo.name = "Thumbnail";
        thumbGo.transform.SetParent(root.transform, false);
        thumbGo.transform.localScale = new Vector3(panelWidth, panelHeight, 1f);

        // Custom/UnlitDoubleSided is a project-local shader asset (always in build).
        // It renders both faces so panels are visible from any angle — this avoids
        // the single-sided issue where panels disappear when viewed from behind.
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

        // Border quad (slightly larger, behind thumbnail) — this is the frame that shows
        // the session's live risk color (SessionThumbnailPanel drives it from
        // SimPub/Status/risk). It MUST carry an explicit shader-based material: leaving the
        // default primitive material meant the risk-color MaterialPropertyBlock never rendered
        // (magenta/uncolored under URP on Android). Mirrors InterventionRgbPanelSpawner's
        // border-material fix; borderMaterial can't be Inspector-assigned since these panels
        // are built at runtime.
        var borderGo = GameObject.CreatePrimitive(PrimitiveType.Quad);
        borderGo.name = "Border";
        borderGo.transform.SetParent(root.transform, false);
        borderGo.transform.localScale  = new Vector3(panelWidth + 0.024f, panelHeight + 0.024f, 1f);
        // NEGATIVE local Z: the selector panel's +Z faces the viewer (LookRotation toward
        // head), so the border must sit BEHIND the thumbnail (−Z, away from viewer). With the
        // double-sided UnlitDoubleSided material, a +Z border would render its rear face on top
        // of the feed and cover it entirely — only the 6 mm rim should show as the risk frame.
        // (The RGB diamond uses +Z because its 180° facing flip points local +Z away from the viewer.)
        borderGo.transform.localPosition = new Vector3(0f, 0f, -0.002f);
        Material borderMat;
        if (borderMaterial != null)
        {
            borderMat = borderMaterial;
        }
        else if (guaranteedShader != null)
        {
            borderMat = new Material(guaranteedShader);
            borderMat.color = noSignalColor;
            borderMat.SetColor("_Color", noSignalColor);
        }
        else
        {
            borderMat = borderGo.GetComponent<Renderer>().material;
        }
        borderGo.GetComponent<Renderer>().material = borderMat;
        Destroy(borderGo.GetComponent<Collider>());

        // Label canvas
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
        txt.text      = $"Session {index}";

        // LIVE indicator (red dot + "LIVE" text), top-center — shown only on the panel matching
        // the currently active single-view session (combined view). Hidden by default;
        // SessionThumbnailPanel.Update() toggles it via SessionRegistry comparison.
        //
        // CRITICAL: mount it at POSITIVE local Z (toward the viewer). The panel's +Z faces the
        // head (LookRotation), and the thumbnail quad sits at z=0. Anything at negative z is
        // BEHIND the opaque thumbnail and is occluded head-on — the earlier −0.003 placement is
        // why it only flashed into view when a panel rotated edge-on during a Y-reposition.
        // Text uses TextMesh + builtin Arial (the proven-on-Android pattern from RiskBarChart);
        // UnityEngine.UI.Text with no font assigned renders nothing on Quest.
        var liveGo = new GameObject("LiveIndicator");
        liveGo.transform.SetParent(root.transform, false);
        // Slight -x shift so the whole "● LIVE" group (dot to the viewer-left of the wide text)
        // stays visually centered on the panel top.
        liveGo.transform.localPosition = new Vector3(-0.012f, panelHeight * 0.5f - 0.028f, 0.012f);
        liveGo.SetActive(false);

        // Dot on the VIEWER'S LEFT (start of "LIVE"). The panel faces the viewer with +Z, so
        // local +X = viewer's left; the dot therefore goes at POSITIVE local x while the flipped
        // text sits near center/negative-x — giving "● LIVE" from the viewer's side. The "LIVE"
        // mesh is ~0.07 wide (its "L" reaches ~+0.030 local x), so the dot sits past that (+0.058)
        // to land BEFORE the L with a clear SPACE ("● LIVE"), not touching it ("●LIVE").
        var dotGo = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        dotGo.name = "LiveDot";
        dotGo.transform.SetParent(liveGo.transform, false);
        dotGo.transform.localPosition = new Vector3(0.058f, 0f, 0f);
        dotGo.transform.localScale    = Vector3.one * 0.014f;
        Destroy(dotGo.GetComponent<Collider>());
        var dotRenderer = dotGo.GetComponent<Renderer>();
        if (guaranteedShader != null)
        {
            var dotMat = new Material(guaranteedShader);
            dotMat.color = Color.red;
            dotMat.SetColor("_Color", Color.red);
            dotMat.SetColor("_BaseColor", Color.red);
            dotRenderer.material = dotMat;
        }
        dotRenderer.shadowCastingMode    = UnityEngine.Rendering.ShadowCastingMode.Off;
        dotRenderer.receiveShadows       = false;

        var liveTextGo = new GameObject("LiveText");
        liveTextGo.transform.SetParent(liveGo.transform, false);
        liveTextGo.transform.localPosition = new Vector3(-0.006f, 0f, 0f);
        // The panel faces the viewer with its +Z (LookRotation toward head), so a TextMesh is
        // read from its back side and appears horizontally mirrored. Negate local X to flip it
        // upright. The builtin font shader (GUI/Text Shader) is Cull Off, so the flipped winding
        // still renders. MiddleCenter anchor keeps it centered at localPosition under the flip.
        liveTextGo.transform.localScale = new Vector3(-1f, 1f, 1f);
        var liveTm = liveTextGo.AddComponent<TextMesh>();
        var liveFont = Resources.GetBuiltinResource<Font>("Arial.ttf");
        if (liveFont != null) liveTm.font = liveFont;
        liveTm.text          = "LIVE";
        liveTm.fontSize      = 46;
        liveTm.fontStyle     = FontStyle.Bold;
        liveTm.characterSize = 0.011f;
        liveTm.anchor        = TextAnchor.MiddleCenter;
        liveTm.alignment     = TextAlignment.Center;
        liveTm.color         = Color.red;
        var liveTmMr = liveTextGo.GetComponent<MeshRenderer>();
        if (liveTmMr != null)
        {
            if (liveFont != null) liveTmMr.sharedMaterial = liveFont.material;
            liveTmMr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            liveTmMr.receiveShadows    = false;
        }

        // PAUSED badge (amber "PAUSED" text), BOTTOM-center — shown when the session reports
        // sim-paused on SimPub/Status/paused. Positioned opposite the top-center LIVE badge so a
        // session that is both LIVE and PAUSED shows both without overlap. Same +z-toward-viewer
        // placement as LIVE (negative z would be occluded behind the opaque thumbnail head-on).
        // Hidden by default; SessionThumbnailPanel.Update() toggles it from the paused flag.
        var pausedGo = new GameObject("PausedIndicator");
        pausedGo.transform.SetParent(root.transform, false);
        // Centered on the panel bottom.
        pausedGo.transform.localPosition = new Vector3(0f, -(panelHeight * 0.5f - 0.028f), 0.012f);
        pausedGo.SetActive(false);

        Color pausedColor = new Color(1f, 0.75f, 0f, 1f); // amber, distinct from red LIVE

        var pausedTextGo = new GameObject("PausedText");
        pausedTextGo.transform.SetParent(pausedGo.transform, false);
        pausedTextGo.transform.localPosition = Vector3.zero;
        // Panel faces the viewer with +Z, so a TextMesh reads mirrored from the back — negate
        // local X to flip it upright (same as LiveText). Builtin font shader is Cull Off.
        pausedTextGo.transform.localScale = new Vector3(-1f, 1f, 1f);
        var pausedTm = pausedTextGo.AddComponent<TextMesh>();
        var pausedFont = Resources.GetBuiltinResource<Font>("Arial.ttf");
        if (pausedFont != null) pausedTm.font = pausedFont;
        pausedTm.text          = "❚❚ PAUSED";
        pausedTm.fontSize      = 46;
        pausedTm.fontStyle     = FontStyle.Bold;
        // Small so it reads cleanly at the panel bottom without crowding the thumbnail.
        pausedTm.characterSize = 0.006f;
        pausedTm.anchor        = TextAnchor.MiddleCenter;
        pausedTm.alignment     = TextAlignment.Center;
        pausedTm.color         = pausedColor;
        var pausedTmMr = pausedTextGo.GetComponent<MeshRenderer>();
        if (pausedTmMr != null)
        {
            if (pausedFont != null) pausedTmMr.sharedMaterial = pausedFont.material;
            pausedTmMr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            pausedTmMr.receiveShadows    = false;
        }

        // Collider on root for ray-cast
        var col = root.AddComponent<BoxCollider>();
        col.size   = new Vector3(panelWidth, panelHeight, 0.01f);
        col.center = Vector3.zero;

        // Panel component (configured after spawn)
        var panel = root.AddComponent<SessionThumbnailPanel>();
        panel.borderRenderer    = borderGo.GetComponent<Renderer>();
        panel.thumbnailImage    = null; // RawImage not used when using Renderer
        panel.labelText         = txt;
        // Refs the panel toggles to hide itself until its session publishes a frame
        // (reveal-on-first-frame). Hiding visuals + collider keeps the ZMQ subscriber
        // alive, unlike SetActive(false)/enabled=false which would kill it.
        panel.thumbnailRenderer = thumbGo.GetComponent<Renderer>();
        panel.panelCollider     = col;
        panel.liveIndicatorRoot = liveGo;
        panel.pausedIndicatorRoot = pausedGo;
        panel.pausedIndicatorText = pausedTm;

        // Wire thumbnail material for texture (RawImage-free path):
        // We'll update the thumbnail quad's main texture in the panel's Update
        // via a custom bridge component.
        var bridge = root.AddComponent<ThumbnailRendererBridge>();
        bridge.thumbnailRenderer = thumbGo.GetComponent<Renderer>();
        bridge.panel             = panel;

        return root;
    }

    // --- OOD operator-override bookkeeping -----------------------------------
    // What this supervisor last COMMANDED for a session, so an observed state that differs
    // can be attributed to a human rather than to itself.
    private readonly Dictionary<int, bool>  _oodCommandedPaused = new Dictionary<int, bool>();
    private readonly Dictionary<int, float> _oodManualOverrideUntil = new Dictionary<int, float>();
    private readonly HashSet<int>           _oodSkipLogged = new HashSet<int>();
    private bool _loggedOodDisabled;
    private bool _loggedOodNotOwner;

    [Tooltip("After the operator manually pauses/resumes a session, OOD automation leaves "
           + "that session alone for this long. Without it, a manual unpause was undone "
           + "within the supervisor's 2 s re-assert interval.")]
    public float oodManualOverrideSeconds = 15.0f;

    /// <summary>Has a human changed this session's transport recently?
    ///
    /// Detected by comparing the observed paused state against what this supervisor last
    /// commanded: any divergence it did not cause is an operator action.</summary>
    private bool ManualOverrideActive(SessionThumbnailPanel panel, float now)
    {
        int idx = panel.sessionIndex;
        if (_oodManualOverrideUntil.TryGetValue(idx, out float until) && now < until)
            return true;

        if (_oodCommandedPaused.TryGetValue(idx, out bool commandedPaused)
            && panel.IsPaused != commandedPaused)
        {
            _oodManualOverrideUntil[idx] = now + Mathf.Max(0f, oodManualOverrideSeconds);
            _oodCommandedPaused.Remove(idx);
            Debug.Log($"[MultiSessionGridManager] OOD: operator changed S{idx} "
                    + $"(observed paused={panel.IsPaused}, we commanded {commandedPaused}) — "
                    + $"backing off for {oodManualOverrideSeconds:F0}s.");
            return true;
        }
        return false;
    }

    private void LogOodSkipOnce(int sessionIndex, string reason)
    {
        if (_oodSkipLogged.Add(sessionIndex))
            Debug.Log($"[MultiSessionGridManager] OOD: skipping S{sessionIndex} ({reason}).");
    }

    /// <summary>True when this device is the OOD arbiter.
    ///
    /// The launcher derives ownership from START_VR (headset driving -> "quest", PC grid
    /// driving -> "desktop") and advertises it in the discovery beacon. Exactly one side
    /// may send PAUSE: two supervisors targeting the same sessions fight each other and
    /// the operator. An older launcher omits the field, leaving this false — OOD stays
    /// off, which is the safe direction.</summary>
    private bool OodOwnedHere()
    {
        string owner = PublisherDiscoveryListener.LastOodOwner;
        if (string.IsNullOrEmpty(owner))
            return false;
        return owner.Equals("quest", StringComparison.OrdinalIgnoreCase);
    }

    private void EnforceOodActiveCellLimit()
    {
        if (!enableOodActiveCellLimit || !_gridSpawned || _transitioning)
            return;

        // Explicit opt-in from the launcher beacon. "Disabled" must be a flag, not a side
        // effect of the -1.0 threshold sentinel (which, since risk is always >= 0, actually
        // means "every session is OOD" -- the most aggressive setting possible).
        if (!PublisherDiscoveryListener.LastOodEnabled)
        {
            if (!_loggedOodDisabled)
            {
                _loggedOodDisabled = true;
                Debug.Log("[MultiSessionGridManager] OOD auto-pause DISABLED "
                        + "(launcher did not advertise ood_enabled).");
            }
            return;
        }

        // GUARD 1 — the feature is off unless a threshold was actually configured.
        // A negative threshold is the explicit disabled sentinel. Since risk is always
        // >= 0, treating it as a normal comparison would classify every session as OOD.
        if (accOodThreshold < 0f)
        {
            if (!_loggedOodDisabled)
            {
                _loggedOodDisabled = true;
                Debug.Log($"[MultiSessionGridManager] OOD auto-pause DISABLED "
                        + $"(accOodThreshold={accOodThreshold:F2} < 0). Set a threshold >= 0 to enable.");
            }
            return;
        }
        if (!OodOwnedHere())
        {
            if (!_loggedOodNotOwner)
            {
                _loggedOodNotOwner = true;
                Debug.Log($"[MultiSessionGridManager] OOD auto-pause not owned by this device "
                        + $"(beacon ood_owner='{PublisherDiscoveryListener.LastOodOwner}'); "
                        + "the desktop grid arbitrates.");
            }
            return;
        }

        float now = Time.unscaledTime;
        if (now < _nextOodEvaluateAt)
            return;
        _nextOodEvaluateAt = now + Mathf.Max(0.1f, oodEvaluateIntervalSeconds);

        int activeLimit = Mathf.Max(0, maxActiveOodCells);
        var oodPanels = new List<SessionThumbnailPanel>();
        foreach (var p in _panels)
        {
            if (p == null || !p.IsRevealed)
                continue;
            if (p.RiskValue >= accOodThreshold)
                oodPanels.Add(p);
        }

        oodPanels.Sort((a, b) => b.RiskValue.CompareTo(a.RiskValue));
        var keepActive = new HashSet<int>();
        for (int i = 0; i < Mathf.Min(activeLimit, oodPanels.Count); i++)
            keepActive.Add(oodPanels[i].sessionIndex);

        var stillOodPaused = new HashSet<int>();
        for (int i = 0; i < oodPanels.Count; i++)
        {
            var panel = oodPanels[i];
            bool shouldPauseForOod = !keepActive.Contains(panel.sessionIndex);
            if (!shouldPauseForOod)
                continue;

            // GUARD 2 — never touch the session the operator is working in. Without this
            // the supervisor re-paused the single-view session every 2 s, so a manual
            // unpause was undone before the operator could do anything with it, and
            // intervention was effectively impossible.
            if (SessionRegistry.HasSelection
                && panel.sessionIndex == SessionRegistry.SelectedSessionIndex)
            {
                panel.SetOodPauseReason(false);
                LogOodSkipOnce(panel.sessionIndex, "in single view");
                continue;
            }

            // GUARD 3 — never interrupt an intervention, on ANY session.
            // This used to test `InterventionStatusHud.InterventionActive && HasSelection &&
            // index == SelectedSessionIndex`, which is strictly implied by Guard 2 above — so
            // it was dead code and an intervention on a NON-selected session (e.g. started
            // from the desktop grid, or still finishing after the operator exited to the
            // selector) was left unprotected and could be force-paused mid-takeover.
            // panel.IsIntervening is a per-session level republished at ~4 Hz by that
            // session's own runtime, so it is correct regardless of what is selected here.
            if (panel.IsIntervening || InterventionStatusHud.InterventionActive)
            {
                panel.SetOodPauseReason(false);
                LogOodSkipOnce(panel.sessionIndex, "intervention active");
                continue;
            }

            // GUARD 4 — an operator pause/resume wins for a while. If the observed paused
            // state is not what we last commanded, a human changed it; back off instead of
            // immediately overriding them.
            if (ManualOverrideActive(panel, now))
            {
                panel.SetOodPauseReason(false);
                LogOodSkipOnce(panel.sessionIndex, "manual override");
                continue;
            }

            stillOodPaused.Add(panel.sessionIndex);
            panel.SetOodPauseReason(true);
            _lastOodPauseCommandAt.TryGetValue(panel.sessionIndex, out float lastPauseCommandAt);
            if (!panel.IsPaused && now - lastPauseCommandAt >= 2.0f)
            {
                SessionCommandSender.Send(panel.publisherIp, panel.commandPort, "PAUSE_AUTO");
                _lastOodPauseCommandAt[panel.sessionIndex] = now;
                _oodCommandedPaused[panel.sessionIndex] = true;
                _oodSkipLogged.Remove(panel.sessionIndex);
                Debug.Log($"[MultiSessionGridManager] OOD limit: PAUSE S{panel.sessionIndex} "
                        + $"acc={panel.RiskValue:F2} threshold={accOodThreshold:F2} "
                        + $"activeLimit={activeLimit}");
            }
        }

        var toClear = new List<int>();
        foreach (int sessionIndex in _oodPausedSessions)
        {
            if (stillOodPaused.Contains(sessionIndex))
                continue;
            toClear.Add(sessionIndex);
        }

        foreach (int sessionIndex in toClear)
        {
            _oodPausedSessions.Remove(sessionIndex);
            _lastOodPauseCommandAt.Remove(sessionIndex);
            var panel = _panels.Find(p => p != null && p.sessionIndex == sessionIndex);
            if (panel == null)
                continue;
            panel.SetOodPauseReason(false);
            // Do not auto-resume a session the operator paused themselves: only undo a
            // pause this supervisor issued.
            bool weParkedIt = _oodCommandedPaused.TryGetValue(sessionIndex, out bool cp) && cp;
            if (panel.IsPaused && weParkedIt && !panel.IsIntervening
                && !ManualOverrideActive(panel, now))
            {
                SessionCommandSender.Send(panel.publisherIp, panel.commandPort, "RESUME_AUTO");
                _oodCommandedPaused[sessionIndex] = false;
                Debug.Log($"[MultiSessionGridManager] OOD limit: RESUME S{panel.sessionIndex} "
                        + $"acc={panel.RiskValue:F2} threshold={accOodThreshold:F2} "
                        + $"activeLimit={activeLimit}");
            }
            else if (panel.IsPaused && !weParkedIt)
            {
                Debug.Log($"[MultiSessionGridManager] OOD: leaving S{sessionIndex} paused "
                        + "(operator paused it, not us).");
            }
        }

        _oodPausedSessions.Clear();
        foreach (int sessionIndex in stillOodPaused)
            _oodPausedSessions.Add(sessionIndex);
    }

    // -----------------------------------------------------------------------
    // Ray interaction

    private void UpdateRay()
    {
        // Get controller ray in world space
        if (!TryGetControllerRay(out Vector3 origin, out Vector3 direction))
        {
            ClearHover();
            return;
        }

        // Cast against panel colliders. A SphereCast (thicker pointer) makes grazing hits count so
        // the tightly-spaced panels are easier to target; radius 0 falls back to a thin raycast.
        SessionThumbnailPanel hit = null;
        RaycastHit info;
        bool gotHit = hoverSphereCastRadius > 0f
            ? Physics.SphereCast(origin, hoverSphereCastRadius, direction, out info, rayMaxDistance)
            : Physics.Raycast(origin, direction, out info, rayMaxDistance);
        if (gotHit)
        {
            hit = info.collider.GetComponentInParent<SessionThumbnailPanel>();
        }

        // Publish the ray for ControllerRayVisual. Done here, after the cast, so the beam is
        // drawn from the same smoothed direction and ends at the same hit point that decides
        // hover — no second cast, no divergence between what is drawn and what is selectable.
        RayOrigin    = origin;
        RayDirection = direction;
        RayHasHit    = gotHit;
        RayHitPoint  = gotHit ? info.point : origin + direction * rayMaxDistance;
        RayFrame     = Time.frameCount;

        // Update hover state
        if (hit != _hoveredPanel)
        {
            if (_hoveredPanel != null) _hoveredPanel.SetHover(false);
            _hoveredPanel = hit;
            if (_hoveredPanel != null)
            {
                _hoveredPanel.SetHover(true);
                // Pre-connect the command socket now so the ENTER_SINGLE sent on
                // trigger-select finds a warm pipe instead of timing out on a cold
                // socket (the cause of dropped ENTER_SINGLE → no point clouds).
                SessionCommandSender.Warm(_hoveredPanel.publisherIp, _hoveredPanel.commandPort);
            }
        }

        // Combined view: InterventionBackButton (in the other, simultaneously-loaded scene)
        // reads this to suppress its own exit-trigger fire on the same RIGHT index trigger.
        IsHoveringSelectablePanel = _hoveredPanel != null && _hoveredPanel.IsRevealed;

        // Trigger to select (controller trigger on device; EditorTestHelper handles
        // mouse-click selection in editor, so we skip OVRInput there).
#if !UNITY_EDITOR
        if (_hoveredPanel != null && OVRInput.GetDown(OVRInput.RawButton.RIndexTrigger))
        {
            // While the PC anchor is being nudged, ALL command forwarding is suppressed
            // and grid-select is a new command source and honors the same rule.
            if (CombinedViewEnabled && SceneAnchorManager.AdjustModeActive)
                return;

            // MC condition: during a live intervention the right controller IS the arm.
            // This trigger is the MC clutch, not a session switch — a pull with the ray
            // incidentally crossing a panel would otherwise switch sessions mid-takeover
            // (and, via the deselect auto-cancel, abort the takeover the operator is in the
            // middle of). Selection returns the moment they ACCEPT (Left X) or REJECT
            // (Left grip); both are left-hand inputs and stay live throughout.
            //
            // Deliberately NOT gated on InterventionActive alone: in the KT and FACTR
            // conditions the right controller drives nothing, so switching sessions there
            // should still work and still auto-cancel.
            if (MotionControllerModeManager.McControlsLocked)
            {
                Debug.Log("[MultiSessionGridManager] select ignored: MC intervention holds the "
                          + "right controller. Accept (Left X) or reject (Left grip) first.");
                return;
            }

            ExitAllOtherSessions(_hoveredPanel, "manual-select");
            _hoveredPanel.Select();
            _pendingEnterIp = _hoveredPanel.publisherIp;
            _pendingEnterPort = _hoveredPanel.commandPort;
            bool enterSent = SessionCommandSender.Send(_pendingEnterIp, _pendingEnterPort, "ENTER_SINGLE");
            _transitioning = true;
            Debug.Log(
                $"[MultiSessionGridManager] Selected session {_hoveredPanel.sessionIndex} → "
                + $"ENTER_SINGLE result={enterSent} then loading {interventionSceneName}"
            );
            StartCoroutine(ShutdownAndLoad(interventionSceneName));
            return;
        }

        // Button routing in the selector scene. Start/pause is on the RIGHT controller —
        // the same hand as the pointer — so any hovered scene can be paused without
        // entering it:
        //   B tap  → start/pause the HOVERED panel only (Python maps B→sim_toggle)
        //   B hold → start/pause ALL live panels (start/stop everything at once)
        //   B with nothing hovered → not ours; InterventionButtonForwarder pauses the
        //                            session currently in single view instead
        //   Y → REMOVED. Reset is no longer exposed on the headset; Python still accepts
        //       RESET/RESTART for the desktop 'R' key and send_multi_window_command.py
        //   A → ignored (mirroring is single-view only)
        //   X → ignored (intervention/replan is single-view only)
        //   RIGHT grip → HandleReposition (grid relocation), does NOT forward
        //
        // All of the above are RIGHT-hand inputs (A and B), so they are suppressed for the
        // duration of an MC intervention along with select, above. RIGHT A in particular is
        // the MC gripper button — routing it into the grid while the operator is closing the
        // gripper is the same class of bug as the stray select.
        if (MotionControllerModeManager.McControlsLocked) return;

        bool a = OVRInput.GetDown(OVRInput.RawButton.A);
        bool bDown = OVRInput.GetDown(OVRInput.RawButton.B);
        bool bHeld = OVRInput.Get(OVRInput.RawButton.B);
        bool bUp = OVRInput.GetUp(OVRInput.RawButton.B);
        bool x = OVRInput.GetDown(OVRInput.RawButton.X);
        if (a || bDown || bUp || x)
        {
            string pressed = (a ? "A(ignored)" : "") + (bDown ? "B(down)" : "") + (bUp ? "B(up)" : "")
                           + (x ? "X(ignored)" : "");
            // NO sticky last-hovered fallback. It meant that sliding the ray off the grid
            // and pressing B paused whatever panel you last happened to graze — possibly
            // from minutes earlier. With nothing hovered the grid stays out of the way
            // entirely and InterventionButtonForwarder routes B to the session you are
            // actually inside (see its CombinedViewEnabled/IsHoveringSelectablePanel gate).
            var target = _hoveredPanel;
            string targetStr = target != null
                ? $"S{target.sessionIndex} {target.publisherIp}:{target.commandPort}"
                : "<none>";
            Debug.Log(
                $"[MultiSessionGridManager] press={pressed} hovered={(_hoveredPanel != null)} "
                + $"target={targetStr} panels={_panels.Count}"
            );

            if (a)
                Debug.Log("[MultiSessionGridManager] A ignored in selector; mirroring is single-view only.");
            // B = start/pause (tap = hovered, hold = all). Armed ONLY while a panel is
            // hovered, so neither the tap nor the hold-all can fire off an empty ray.
            if (bDown && target != null)
            {
                _bPressPending = true;
                _bBroadcastSent = false;
                _bPressStartTime = Time.unscaledTime;
                _bPressTarget = target;
            }
            if (bUp && _bPressPending)
            {
                if (!_bBroadcastSent && _bPressTarget != null)
                    SessionCommandSender.Send(_bPressTarget.publisherIp, _bPressTarget.commandPort, "SIM_TOGGLE");
                _bPressPending = false;
                _bBroadcastSent = false;
                _bPressTarget = null;
            }
            if (x)
            {
                Debug.Log("[MultiSessionGridManager] X ignored in selector; intervention/replan is single-view only.");
            }
        }
        if (bHeld && _bPressPending && !_bBroadcastSent
            && Time.unscaledTime - _bPressStartTime >= Mathf.Max(0.1f, resetAllHoldSeconds))
        {
            BroadcastToLivePanels("SIM_TOGGLE", "B hold pause/resume-all");
            _bBroadcastSent = true;
        }
#endif
    }

    private int BroadcastToLivePanels(string command, string reason)
    {
        int sent = 0;
        foreach (var p in _panels)
        {
            if (p == null || !p.IsRevealed) continue;
            SessionCommandSender.Send(p.publisherIp, p.commandPort, command);
            sent++;
        }
        Debug.Log($"[MultiSessionGridManager] {reason} broadcast '{command}' to {sent}/{_panels.Count} live panels.");
        return sent;
    }

    private bool TryGetControllerRay(out Vector3 origin, out Vector3 direction)
    {
        origin    = Vector3.zero;
        direction = Vector3.forward;

#if UNITY_EDITOR
        // EditorTestHelper (Awake order -500) creates Camera.main before this runs.
        // Poll its ray directly for accurate hover/select.
        var editorHelper = GetComponent<EditorTestHelper>();
        if (editorHelper != null && editorHelper.hasEditorRay)
        {
            origin    = editorHelper.editorRayOrigin;
            direction = editorHelper.editorRayDirection;
            return true;
        }
        // Fallback: use Camera.main (which EditorTestHelper already set as the clean camera)
        if (Camera.main != null)
        {
            origin    = Camera.main.transform.position;
            direction = Camera.main.transform.forward;
            return true;
        }
        return false;
#else
        if (_trackingSpace == null)
            _trackingSpace = FindTrackingSpace();

        // OVRInput gives local-to-TrackingSpace position/rotation.
        // RIGHT controller is the selector pointer (RIndexTrigger selects).
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

        if (direction.sqrMagnitude <= 0.01f)
            return false;

        // Low-pass the ray direction to take the twitch out of the raw controller pose so panels
        // are easier to hold/target. Slerp(raw, previous, raySmoothing): 0 = raw, higher = steadier.
        direction = direction.normalized;
        if (_smoothedRayDir == Vector3.zero)
            _smoothedRayDir = direction;             // first frame: seed, no lag
        else
            _smoothedRayDir = Vector3.Slerp(direction, _smoothedRayDir, Mathf.Clamp01(raySmoothing));
        direction = _smoothedRayDir;
        return true;
#endif
    }

    private void ClearHover()
    {
        IsHoveringSelectablePanel = false;
        if (_hoveredPanel == null) return;
        _hoveredPanel.SetHover(false);
        _hoveredPanel = null;
    }

    private Transform FindTrackingSpace()
    {
        var go = GameObject.Find(trackingSpaceName);
        return go != null ? go.transform : null;
    }

    // -----------------------------------------------------------------------
    // Scene transition — graceful NetMQ shutdown before loading

    /// <summary>
    /// Local NetMQ teardown before loading the next scene.
    ///
    /// The old path called NetMQConfig.Cleanup(false) on every scene switch.
    /// That avoided stuck pollers, but it also terminated contexts that the next
    /// scene needed to recreate immediately. Keep global cleanup as an explicit
    /// fallback only; normal transitions close the sender and panels locally.
    ///
    /// Mirror of InterventionBackButton.CleanupAndBack().
    /// </summary>
    private System.Collections.IEnumerator ShutdownAndLoad(string sceneName)
    {
        // Resend ENTER_SINGLE on the now-warm socket before tearing it down. If the
        // first send (on select) hit a still-connecting socket and timed out, this
        // retry lands so Python flips single_view_active → point clouds + single-view
        // mirror work. enter_single_view() is idempotent on the Python side.
        if (!string.IsNullOrEmpty(_pendingEnterIp))
        {
            yield return null; // let the warm socket finish connecting
            bool resent = SessionCommandSender.Send(_pendingEnterIp, _pendingEnterPort, "ENTER_SINGLE");
            Debug.Log($"[MultiSessionGridManager] ENTER_SINGLE resend result={resent} -> {_pendingEnterIp}:{_pendingEnterPort}");
            _pendingEnterIp = null;
        }

        yield return new WaitForSecondsRealtime(Mathf.Max(0.0f, transitionCommandFlushSeconds));

        if (CombinedViewEnabled)
        {
            // Combined view: the grid stays loaded and selectable, so none of the
            // panel-disable / SessionCommandSender teardown below applies — that would
            // kill the other 14 panels' warm sockets and hover-connect cache for no reason.
            var existing = SceneManager.GetSceneByName(sceneName);
            if (existing.IsValid() && existing.isLoaded)
            {
                // Already in single view, switching sessions. In RGB mode, re-point the existing
                // scene IN PLACE (no reload): the ENTER_SINGLE above already flushed, so the new
                // session's active-only cams are warming up. This avoids the unload+reload churn
                // that made the diamond panels lag / stick on the old scene / lose feed while the
                // 15 grid subscribers stayed live (their pollers race the reload teardown). The
                // anchor/MujocoScene stays put, so panel positions don't flicker either.
                if (SessionRegistry.SelectedRgbMode && InterventionSessionBootstrap.Instance != null)
                {
                    InterventionSessionBootstrap.Instance.ReconfigureToSelectedSession();
                    Debug.Log("[MultiSessionGridManager] Combined RGB switch → reconfigured single view in place (no reload).");
                    _transitioning = false;
                    yield break;
                }

                // Point-cloud mode (or no live bootstrap): drop the old InterventionV1 instance
                // before loading a fresh one, so InterventionSessionBootstrap's Awake-time rewiring
                // re-runs cleanly for the PC loaders.
                yield return SceneManager.UnloadSceneAsync(existing);
            }
            yield return SceneManager.LoadSceneAsync(sceneName, LoadSceneMode.Additive);

            // Resume ray/selection processing now that the load has settled — this is
            // what keeps the grid selectable immediately, including right after a switch.
            _transitioning = false;
            yield break;
        }

        // --- Exclusive-swap path (CombinedViewEnabled == false), unchanged ---

        // 1) Fallback-only global cleanup. The normal path uses local socket
        //    cleanup so the next scene can create fresh NetMQ subscribers.
        if (forceGlobalNetMqCleanupOnSceneSwitch)
        {
            try { NetMQ.NetMQConfig.Cleanup(false); } catch { }
        }

        // 2) Tear down our static NetMQ state (push sockets + monitor poller).
        try { SessionCommandSender.Shutdown(); } catch { }

        // 3) Disable panels — their OnDisable→Shutdown completes without
        //    taking the whole NetMQ context down.
        foreach (var p in _panels)
        {
            if (p != null && p.enabled)
                p.enabled = false;
        }

        // 4) Yield a couple of frames so OnDisable callbacks finish on the main
        //    thread before SceneManager.LoadScene begins destroying GameObjects.
        yield return null;
        yield return null;
        SceneManager.LoadScene(sceneName);
    }

    // -----------------------------------------------------------------------
    // Cleanup

    void OnDestroy()
    {
        foreach (var p in _panels)
        {
            if (p != null) p.SetHover(false);
        }
        _panels.Clear();
    }
}

// ThumbnailRendererBridge lives in ThumbnailRendererBridge.cs
// (Unity requires each MonoBehaviour to have its own file).
