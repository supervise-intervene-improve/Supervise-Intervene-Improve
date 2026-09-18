using UnityEngine;

/// <summary>
/// Runs in SIIScene_InterventionV1 BEFORE all subscriber Start() calls.
/// Reads the session selected in SIIScene_SessionSelector (via SessionRegistry /
/// PlayerPrefs) and reconfigures every ZMQ subscriber component to use that
/// session's IP and port.
///
/// Sessions stay in PURE THUMBNAIL MODE for the whole app lifetime:
///   - Python runs with --no_mujoco_publisher → no scene mesh, no robot, no wrist cam
///   - Only the point cloud (top/right/left depth cameras) renders in single view
///   - SimPubClient is disabled (no scene service to talk to)
///   - PC anchors to the "MujocoScene" GameObject spawned by SceneAnchorManager
///     (controller-tunable runtime pose, persisted to PlayerPrefs)
///
/// PROMOTE/DEMOTE was tried for in-process MujocoPublisher activation and
/// proved unworkable (DEMOTE never delivered, conflicting publishers, scene
/// corruption). Dropped entirely. The hooks remain in runtime_impl.py as
/// dead code in case the design is revisited.
/// </summary>
[DefaultExecutionOrder(-10000)]
public class InterventionSessionBootstrap : MonoBehaviour
{
    [Header("Debug")]
    public bool logReconfiguration = true;

    [Header("Single View Ready")]
    [Tooltip("Seconds to wait after scene startup before asking Python for a fresh point-cloud burst.")]
    public float singleViewReadyDelaySeconds = 0.35f;

    [Header("Risk Bar Chart")]
    [Tooltip("Attach the single-view RiskBarChart HUD. Default OFF: with the grid visible "
           + "behind the point cloud in combined view, the grid's own live thumbnails + "
           + "risk-colored borders already show every session, making this HUD redundant. "
           + "Kept fully implemented — flip on to bring it back.")]
    public bool enableRiskBarChart = false;

    private bool _hasSelection;
    private int _sessionIndex;
    private string _publisherIp;
    private int _cmdPort;

    /// <summary>The active single-view bootstrap. Used by MultiSessionGridManager to reconfigure
    /// this scene in place on a combined-view session switch (RGB mode) instead of reloading it.</summary>
    public static InterventionSessionBootstrap Instance { get; private set; }

    // Session-cached single-view components captured at spawn, so a combined-view switch can
    // re-point them without a scene reload. (RGB spawner is null in point-cloud mode.)
    private InterventionRgbPanelSpawner _rgbSpawner;
    private InterventionButtonForwarder _buttonForwarder;
    private InterventionStatusHud _statusHud;

    void Awake()
    {
        Instance = this;
        // Combined view: this scene is loaded ADDITIVELY on top of the still-active grid
        // scene. Each scene ships its OWN Meta Building-Block OVR camera rig — a root
        // GameObject "[BuildingBlock] Camera Rig" containing OVRManager + OVRCameraRig and the
        // "TrackingSpace" -> CenterEyeAnchor(MainCamera+AudioListener)/eye/hand-anchor subtree.
        // With BOTH rigs live, the two OVRCameraRigs mis-drive the display/tracking so all
        // world-fixed content (grid panels, point cloud) appears STUCK TO THE LENS / follows
        // the head. The grid itself is world-rooted and correct — the head-lock is purely this
        // duplicate rig.
        //
        // Previous attempt disabled only the Camera+AudioListener components and tried to
        // Destroy a ROOT named "TrackingSpace" — but TrackingSpace is NESTED inside
        // "[BuildingBlock] Camera Rig", never a root, so the destroy silently no-op'd and the
        // OVRCameraRig/anchors stayed live. Fix: find the rig root by CONTENT (the one root
        // whose subtree contains a Camera), SetActive(false) it, then Destroy it.
        //
        // Why this is safe:
        //  - This Bootstrap runs at [DefaultExecutionOrder(-10000)]; OVRManager/OVRCameraRig
        //    declare no execution order (0). So this Awake runs FIRST during the additive load,
        //    and SetActive(false) here means the duplicate rig's OVRManager/OVRCameraRig/Camera
        //    Awakes never run at all — the rig is inert before it can touch tracking.
        //  - Even if it did awake, OVRManager.InitOVRManager() self-destructs a duplicate
        //    (if (instance != null) DestroyImmediate(this)), so the grid scene's OVRManager
        //    (created first at app start) stays the singleton regardless. No singleton hijack.
        //  - Destroying the whole root also removes the second "TrackingSpace", so
        //    GameObject.Find("TrackingSpace") (used for the controller ray) unambiguously
        //    resolves to the grid scene's.
        // Scoped to this scene's own root objects only, never the grid scene's.
        if (MultiSessionGridManager.CombinedViewEnabled)
        {
            var thisScene = gameObject.scene;
            GameObject rigRoot = null;
            foreach (var rootGo in thisScene.GetRootGameObjects())
            {
                // The camera rig is the one root whose subtree owns a Camera (the eye anchors).
                // No other root in SIIScene_InterventionV1 contains a Camera.
                if (rootGo.GetComponentInChildren<Camera>(true) != null)
                {
                    rigRoot = rootGo;
                    break;
                }
            }
            if (rigRoot != null)
            {
                rigRoot.SetActive(false);  // synchronous — stops its OVR Awakes/Updates immediately
                Destroy(rigRoot);          // remove it entirely
            }
            if (logReconfiguration)
                Debug.Log("[InterventionSessionBootstrap] Combined view: destroying duplicate camera rig root "
                        + $"'{(rigRoot != null ? rigRoot.name : "<not found>")}' "
                        + "(grid scene's rig stays the sole active XR rig).");
        }

        SessionRegistry.Load();

        if (!SessionRegistry.HasSelection)
        {
            if (logReconfiguration)
                Debug.Log("[InterventionSessionBootstrap] No session selected — using scene defaults.");
            return;
        }

        int    idx  = SessionRegistry.SelectedSessionIndex;
        int    port = SessionRegistry.SelectedTopicPort;
        string ip   = SessionRegistry.SelectedPublisherIp;
        _hasSelection = true;
        _sessionIndex = idx;
        _publisherIp = ip;
        _cmdPort = port + 5;

        // End-to-end latency probe: point it at the SAME command endpoint the button
        // forwarder uses, so echoes ride the socket SessionCommandSender has already
        // warmed. Inert until the publisher runs with --latency_probe (an unstamped
        // frame fails the magic/checksum test, and the probe goes dormant by itself),
        // so this is safe to arm unconditionally.
        LatencyStampProbe.EchoIp   = _publisherIp;
        LatencyStampProbe.EchoPort = _cmdPort;
        LatencyStampProbe.Enabled  = true;
        LatencyStampProbeDriver.EnsureExists();

        if (logReconfiguration)
            Debug.Log($"[InterventionSessionBootstrap] Reconfiguring for session {idx}: ip={ip} topicPort={port}");

        // --- SimPubRgbdSubscriber (wrist camera panel) ---
        // No robot mesh = no wrist link to anchor to. Disable follow-pose,
        // enable viewer-fallback so any stray frames at least appear somewhere
        // (HUD-style). In practice the launcher doesn't subscribe wrist at
        // higher quality so this is mostly inert, but keep it sane.
        foreach (var c in FindObjectsByType<SimPubRgbdSubscriber>(FindObjectsSortMode.None))
        {
            c.publisherIp = ip;
            c.topicPort   = port;
            c.parentRendererToSimScene = false;
            c.followCameraPose = false;
            c.enableViewerFallbackWhenPoseMissing = true;
            if (logReconfiguration)
                Debug.Log($"[InterventionSessionBootstrap]   SimPubRgbdSubscriber '{c.name}' → {ip}:{port} (thumbnail mode)");
        }

        bool rgbMode = SessionRegistry.SelectedRgbMode;

        // Must be set HERE (execution order -10000), before GpuMergedPointCloudBootstrap's
        // Awake at -500. Disabling that component is not enough: Unity still runs Awake()
        // on a disabled component, and it would spawn an enabled merged loader that opens
        // a /pc subscriber this pass has already walked past.
        GpuMergedPointCloudBootstrap.SuppressBootstrap = rgbMode;

        // --- SimPubPointCloudSubscriber (simple PC) ---
        foreach (var c in FindObjectsByType<SimPubPointCloudSubscriber>(FindObjectsSortMode.None))
        {
            if (rgbMode)
            {
                c.enabled = false;
                if (logReconfiguration)
                    Debug.Log($"[InterventionSessionBootstrap]   SimPubPointCloudSubscriber '{c.name}' → DISABLED (RGB panel mode)");
                continue;
            }
            c.publisherIp = ip;
            c.topicPort   = port;
            if (logReconfiguration)
                Debug.Log($"[InterventionSessionBootstrap]   SimPubPointCloudSubscriber '{c.name}' → {ip}:{port}");
        }

        // --- GpuMergedPointCloudLoader (GPU merged PC) ---
        // parentToSimScene=true so PC anchors to the "MujocoScene" GameObject
        // spawned by SceneAnchorManager below. waitForSceneAnchorBeforeFirstDraw
        // stays false so points render even briefly before the anchor exists.
        // In RGB-panel mode the point cloud pipeline is never engaged Python-side
        // either, so these loaders are explicitly disabled (actively inert, same
        // pattern as the SimPubClient disable above) rather than left unconfigured.
        foreach (var c in FindObjectsByType<GpuMergedPointCloudLoader>(FindObjectsSortMode.None))
        {
            if (rgbMode)
            {
                c.enabled = false;
                if (logReconfiguration)
                    Debug.Log($"[InterventionSessionBootstrap]   GpuMergedPointCloudLoader '{c.name}' → DISABLED (RGB panel mode)");
                continue;
            }
            c.publisherIp = ip;
            c.topicPort   = port;
            c.expectedSessionIndex = SessionRegistry.SelectedSessionIndex;
            c.parentToSimScene = true;
            c.waitForSceneAnchorBeforeFirstDraw = false;
            c.renderMode = PointCloudRenderMode.OverlayDepthWrite;
            if (logReconfiguration)
                Debug.Log($"[InterventionSessionBootstrap]   GpuMergedPointCloudLoader '{c.name}' → {ip}:{port} (anchor to SceneAnchorManager's MujocoScene, renderMode=OverlayDepthWrite)");
        }

        // --- GpuPointCloudSubscriber (legacy GPU subscriber, if present) ---
        foreach (var c in FindObjectsByType<GpuPointCloudSubscriber>(FindObjectsSortMode.None))
        {
            if (rgbMode)
            {
                c.enabled = false;
                if (logReconfiguration)
                    Debug.Log($"[InterventionSessionBootstrap]   GpuPointCloudSubscriber '{c.name}' → DISABLED (RGB panel mode)");
                continue;
            }
            c.publisherIp = ip;
            c.topicPort   = port;
            if (logReconfiguration)
                Debug.Log($"[InterventionSessionBootstrap]   GpuPointCloudSubscriber '{c.name}' → {ip}:{port}");
        }

        // --- SimPubClient (scene mesh downloader) ---
        // Sessions run --no_mujoco_publisher and stay there → there is no
        // scene service to talk to. Disable SimPubClient to prevent it from
        // hammering a dead port with retries.
        foreach (var c in FindObjectsByType<SimPubClient>(FindObjectsSortMode.None))
        {
            c.simpubIp        = ip;
            c.simpubTopicPort = port;
            c.enabled         = false;
            if (logReconfiguration)
                Debug.Log($"[InterventionSessionBootstrap]   SimPubClient '{c.name}' → DISABLED (no MujocoPublisher in thumbnail-only mode)");
        }

        // --- Scene anchor manager: spawns the "MujocoScene" GameObject the
        // PC loader will anchor to. Controller-tunable runtime pose with
        // PlayerPrefs persistence. Future hook: ApplyAlignmentData() will
        // be called by a QR detector when that ships.
        gameObject.AddComponent<SceneAnchorManager>();
        if (logReconfiguration)
            Debug.Log("[InterventionSessionBootstrap]   SceneAnchorManager attached (MujocoScene + controller tuning + PlayerPrefs persistence).");

        // --- GpuMergedPointCloudBootstrap (spawns the merged PC loader) ---
        // Runs at execution order -500 (after our -10000). parentToSimScene=true
        // propagates to the dynamically-created loader, which will find
        // "MujocoScene" via TryAttachToSceneRoot polling once SceneAnchorManager
        // spawns it (Start runs after this Awake).
        //
        // CRITICAL: maxPointsPerSource must match the Python --pc_max_points
        // setting (current launcher default 200000). If it's smaller, the
        // loader DROPS every payload silently with the warning
        // "Dropped 'SimPub/Sensors/X/pc' because declared capacity ..."
        // and the PC never renders. maxCombinedPoints = 3 sources × maxPointsPerSource.
        //
        // In RGB-panel mode this bootstrap is disabled instead — Python never
        // publishes /pc topics in that mode (--rgb_mode never passes --pc), so
        // there is nothing for it to render; leaving it enabled would just be
        // dead weight (and risk confusing "Dropped ..." log spam).
        foreach (var c in FindObjectsByType<GpuMergedPointCloudBootstrap>(FindObjectsSortMode.None))
        {
            if (rgbMode)
            {
                c.enabled = false;
                if (logReconfiguration)
                    Debug.Log($"[InterventionSessionBootstrap]   GpuMergedPointCloudBootstrap '{c.name}' → DISABLED (RGB panel mode)");
                continue;
            }
            c.publisherIp = ip;
            c.topicPort   = port;
            c.expectedSessionIndex = SessionRegistry.SelectedSessionIndex;
            c.parentToSimScene = true;
            c.waitForSceneAnchorBeforeFirstDraw = false;
            // Must stay >= the publisher's --pc_max_points, or every payload whose declared
            // capacity exceeds this is silently DROPPED (lesson 21). The publisher now
            // advertises an adaptive high-water mark clamped to --pc_max_points (80000 at
            // 9 windows), so 200000 keeps a wide safety margin at no cost — the loader's
            // decode buffers are sized from the ACTUAL count, not from this.
            c.maxPointsPerSource = 200000;
            // Sizes the combined upload + GPU buffers: 600000 reserved ~96 MB of managed +
            // GPU memory for the ~11000 points actually in flight. 3 sources x the 80000
            // publisher cap is the true worst case.
            c.maxCombinedPoints  = 240000;
            // Explicit, or the bootstrap silently falls back to OverlayTransparent
            // (ZWrite=0) and the compositor cannot depth-reproject the cloud.
            c.SetRenderModeExplicit(PointCloudRenderMode.OverlayDepthWrite);
            if (logReconfiguration)
                Debug.Log($"[InterventionSessionBootstrap]   GpuMergedPointCloudBootstrap '{c.name}' → {ip}:{port} "
                          + $"(parent to MujocoScene, capacity {c.maxPointsPerSource}/source, {c.maxCombinedPoints} combined, "
                          + "renderMode=OverlayDepthWrite)");
        }

        // --- RGB diamond panel spawner (RGB-panel mode only) ---
        // Mirror of the GpuMergedPointCloudBootstrap branch above: spawns 4 camera
        // panels (top/left/right/wrist) anchored to the same "MujocoScene" object
        // SceneAnchorManager owns, instead of point clouds. Kept in its own
        // component/file so the RGB-panel code path never shares code with the
        // point-cloud path above.
        if (rgbMode)
        {
            var rgbSpawner = gameObject.AddComponent<InterventionRgbPanelSpawner>();
            rgbSpawner.publisherIp = ip;
            rgbSpawner.topicPort   = port;
            _rgbSpawner = rgbSpawner;
            if (logReconfiguration)
                Debug.Log($"[InterventionSessionBootstrap]   InterventionRgbPanelSpawner attached → {ip}:{port} "
                          + "(diamond top/left/right/wrist panels anchored to MujocoScene).");
        }

        // --- A/B/X button forwarder (replay control) ---
        // Sends A=pause/resume, B=reset, X=one-press intervention to the selected session's
        // cmd_port via SessionCommandSender. Pure Unity → Python forwarding;
        // doesn't depend on MetaQuest3 XR-node discovery (which is flaky
        // across multi-session setups).
        _buttonForwarder = gameObject.AddComponent<InterventionButtonForwarder>();
        if (logReconfiguration)
            Debug.Log("[InterventionSessionBootstrap]   InterventionButtonForwarder attached (X=intervene, B=sim, Y=reset, Lgrip=cancel → cmd_port).");

        // --- Intervention status HUD ---
        // Subscribes to SimPub/Status/intervention and shows "Intervention has begun" when the
        // robot releases into human control.
        var statusHud = gameObject.AddComponent<InterventionStatusHud>();
        _statusHud = statusHud;
        if (logReconfiguration)
            Debug.Log("[InterventionSessionBootstrap]   InterventionStatusHud attached (SimPub/Status/intervention).");

        // --- Motion controller ZMQ publisher ---
        // Binds tcp://*:6090 and streams right controller pose + buttons at 120 Hz.
        // Starts DISABLED; MotionControllerModeManager enables/disables it via RIGHT A toggle.
        var mcPub = gameObject.AddComponent<MotionControllerZmqPublisher>();
        mcPub.enabled = false;
        if (logReconfiguration)
            Debug.Log("[InterventionSessionBootstrap]   MotionControllerZmqPublisher attached (tcp://*:6090, starts disabled — RIGHT A to activate).");

        // --- Motion controller mode manager ---
        // Handles RIGHT A toggle between replay mode and motion controller mode.
        // Wires to the publisher and HUD (both already on this GameObject).
        var mcMgr = gameObject.AddComponent<MotionControllerModeManager>();
        mcMgr.SetDependencies(mcPub, statusHud);
        if (logReconfiguration)
            Debug.Log("[InterventionSessionBootstrap]   MotionControllerModeManager attached (RIGHT A = toggle MC mode).");

        // --- Risk bar chart ---
        // Subscribes to SimPub/Status/risk on each running session's topic port and renders
        // a head-relative horizontal bar chart on the right side of the FOV. Right trigger
        // on a bar navigates directly to that session's InterventionV1. Discovery data is
        // read from PublisherDiscoveryListener's static cache (seeded when the selector ran).
        if (enableRiskBarChart && PublisherDiscoveryListener.HasCachedDiscovery)
        {
            var chart = gameObject.AddComponent<RiskBarChart>();
            chart.SetDiscoveryData(
                PublisherDiscoveryListener.LastIp,
                PublisherDiscoveryListener.LastBaseTopicPort,
                PublisherDiscoveryListener.LastPortStep,
                PublisherDiscoveryListener.LastNSessions
            );
            if (logReconfiguration)
                Debug.Log($"[InterventionSessionBootstrap]   RiskBarChart attached "
                        + $"({PublisherDiscoveryListener.LastNSessions} sessions, "
                        + $"ip={PublisherDiscoveryListener.LastIp} "
                        + $"basePort={PublisherDiscoveryListener.LastBaseTopicPort} "
                        + $"step={PublisherDiscoveryListener.LastPortStep}).");
        }
        else if (logReconfiguration)
        {
            if (!enableRiskBarChart)
                Debug.Log("[InterventionSessionBootstrap]   RiskBarChart skipped — disabled (enableRiskBarChart=false). "
                        + "Grid panel live thumbnails + risk-colored borders cover this in combined view.");
            else
                Debug.LogWarning("[InterventionSessionBootstrap]   RiskBarChart skipped — no discovery data cached. "
                               + "Start the multi-session launcher and let the selector receive a beacon first.");
        }
    }

    void Start()
    {
        if (!_hasSelection)
            return;

        StartCoroutine(SendSingleViewReadyAfterDelay());
    }

    private System.Collections.IEnumerator SendSingleViewReadyAfterDelay()
    {
        yield return new WaitForSecondsRealtime(Mathf.Max(0.0f, singleViewReadyDelaySeconds));
        bool sent = SessionCommandSender.Send(_publisherIp, _cmdPort, "SINGLE_VIEW_READY");
        if (logReconfiguration)
        {
            Debug.Log(
                $"[InterventionSessionBootstrap] SINGLE_VIEW_READY send result={sent} for session {_sessionIndex} "
                + $"to {_publisherIp}:{_cmdPort}"
            );
        }
    }

    /// <summary>Re-point this ALREADY-LOADED single-view scene to whatever session is now
    /// selected in SessionRegistry, WITHOUT a scene reload. Called by MultiSessionGridManager on a
    /// combined-view RGB session switch to avoid the unload+reload churn (which caused the diamond
    /// panels to lag / stick / lose feed while 15 grid subscribers stayed live). The anchor /
    /// MujocoScene stays put, so panel positions don't flicker. Point-cloud mode still uses the
    /// reload path (this only reconfigures RGB-diamond + forwarder + status HUD).</summary>
    public void ReconfigureToSelectedSession()
    {
        SessionRegistry.Load();
        if (!SessionRegistry.HasSelection)
        {
            Debug.LogWarning("[InterventionSessionBootstrap] ReconfigureToSelectedSession: no selection — ignored.");
            return;
        }

        string ip   = SessionRegistry.SelectedPublisherIp;
        int    port = SessionRegistry.SelectedTopicPort;
        _hasSelection = true;
        _sessionIndex = SessionRegistry.SelectedSessionIndex;
        _publisherIp  = ip;
        _cmdPort      = port + 5;

        // End-to-end latency probe: point it at the SAME command endpoint the button
        // forwarder uses, so echoes ride the socket SessionCommandSender has already
        // warmed. Inert until the publisher runs with --latency_probe (an unstamped
        // frame fails the magic/checksum test, and the probe goes dormant by itself),
        // so this is safe to arm unconditionally.
        LatencyStampProbe.EchoIp   = _publisherIp;
        LatencyStampProbe.EchoPort = _cmdPort;
        LatencyStampProbe.Enabled  = true;
        LatencyStampProbeDriver.EnsureExists();

        if (_rgbSpawner != null)      _rgbSpawner.ReconfigureAll(ip, port);
        if (_buttonForwarder != null) _buttonForwarder.Reconfigure(ip, port + 5);
        if (_statusHud != null)       _statusHud.Reconfigure(ip, port);

        // Re-point the sensor subscribers too. These were previously left pointing at the
        // FIRST session ever selected, so every subsequent switch leaked a live subscriber
        // onto the old session. Python counts those sockets as peers, and a peer count at
        // or above --active_peer_threshold used to promote that unselected session back to
        // full point-cloud rendering — measured 2026-07-28 as the dominant latency cause
        // ("-> ACTIVE (selected=False, peers=6)").
        foreach (var c in FindObjectsByType<SimPubRgbdSubscriber>(FindObjectsSortMode.None))
            c.Reconfigure(ip, port);

        if (!SessionRegistry.SelectedRgbMode)
        {
            foreach (var c in FindObjectsByType<GpuMergedPointCloudLoader>(FindObjectsSortMode.None))
                c.Reconfigure(ip, port, SessionRegistry.SelectedSessionIndex);
        }

        // Fresh sensor warm-up on the newly-selected session (same as scene-load path).
        StartCoroutine(SendSingleViewReadyAfterDelay());

        if (logReconfiguration)
            Debug.Log($"[InterventionSessionBootstrap] ReconfigureToSelectedSession → session {_sessionIndex} {ip}:{port} "
                    + "(in place, no scene reload).");
    }

    void OnDestroy()
    {
        if (Instance == this) Instance = null;
    }
}
