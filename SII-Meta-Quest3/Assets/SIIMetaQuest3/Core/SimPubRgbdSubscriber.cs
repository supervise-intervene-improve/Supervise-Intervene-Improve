using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using NetMQ;
using NetMQ.Sockets;
using UnityEngine;
using UnityEngine.Rendering;
using IRIS.SceneLoader;
using Stopwatch = System.Diagnostics.Stopwatch;

[Serializable]
public class RgbdHeader
{
    public string cam_name;
    public int width;
    public int height;
    public double timestamp;

    public int rgb_len;
    public int depth_len;
    public int pc_len;

    public CamPoseUnity cam_pose_unity;
}

[Serializable]
public class CamPoseUnity
{
    public float[] pos;
    public float[] quat_xyzw;
}

public enum WristPanelTextureSourceMode
{
    RgbdStream = 0,
    StandaloneRgb = 1,
}

public class SimPubRgbdSubscriber : MonoBehaviour
{
    [Header("Connection")]
    public string publisherIp = "127.0.0.1";
    public int topicPort = 7741;

    [Header("Topics")]
    public string topicPrefix = "SimPub/Sensors"; // base prefix for SimPub/Sensors/<cam>/{rgb,rgbd}
    public string[] cameras = new[] { "wrist", "top", "right", "left" };
    public bool strictCameraFilter = false;

    [Header("Texture Source")]
    public WristPanelTextureSourceMode textureSourceMode = WristPanelTextureSourceMode.StandaloneRgb;
    public string standaloneRgbTopic = "SimPub/Sensors/wrist/rgb";

    [Header("Display")]
    public Renderer[] targetRenderers; // set size=1 and drag your Quad Renderer

    [Header("Auto Panel")]
    public bool autoCreatePanelIfMissing = true;
    public string autoPanelName = "WristCamPanel";
    public bool autoPanelUseUnlitMaterial = true;
    public bool autoPanelFlipY = true;

    [Header("Renderer Compatibility")]
    public bool avoidPointCloudRendererAsRgbdTarget = true;
    public bool forceTextureCompatibleMaterial = true;

    [Header("Debug")]
    public int logFirstMessages = 5;
    public float heartbeatLogIntervalSeconds = 1.0f;

    [Header("Wrist Panel Hardening")]
    public bool forceFollowCameraSettingsOnStart = false;
    public bool resetLegacyRendererBindingsOnStart = false;
    public bool forceDedicatedFollowPanel = false;

    [Header("Reconnect")]
    public bool autoReconnectOnIdle = true;
    public float reconnectIdleSeconds = 8.0f;
    public float reconnectCooldownSeconds = 2.0f;

    [Header("Pose Follow (Optional)")]
    public bool followCameraPose = true;
    public string followCameraName = "wrist";
    public int followRendererIndex = 0;
    public bool followPoseIsSceneLocal = true;
    public bool parentRendererToSimScene = true;
    public string simSceneName = "MujocoScene";
    public bool singleRendererShowsFollowCameraOnly = true;
    public Transform viewerCameraOverride;
    public string preferredViewerCameraName = "CenterEyeAnchor";
    public bool warnOnMultipleMainCameras = true;
    public bool parentPanelUnderAnchorLink = true;
    public string panelAnchorLinkName = "hand";
    public string panelAnchorFallbackLinkName = "link7";
    public Vector3 panelLocalOffsetOnAnchor = new Vector3(-0.0033f, 0.0145f, 0.0564f);
    public Vector3 panelLocalEulerOnAnchor = new Vector3(-156.377f, -3.871002f, 0.1029968f);
    public Vector3 panelOffsetLocal = new Vector3(-0.1022f, -0.1143f, 0.1554f);
    public bool panelOffsetInWorldAxes = true;
    public bool calibrateOffsetFromTargetAtRuntime = false;
    public bool calibrationTargetIsSceneLocal = true;
    public Vector3 calibrationTargetPosition = new Vector3(-0.0892f, 0.5618f, 0.4806f);
    public Vector3 panelEulerOffset = new Vector3(0.0f, 0.0f, 0.0f);
    public bool orientTowardViewer = true;
    public Vector3 viewerFacingEulerOffset = new Vector3(-66.031f, 11.848f, -6.22f);
    public bool flattenViewerFacing = true;
    public bool orientTowardFrontCamera = true;
    public string frontCameraName = "top";
    public Vector3 frontCameraFacingEulerOffset = new Vector3(-66.031f, 11.848f, -6.22f);
    public Vector3 panelScale = new Vector3(0.10f, 0.060f, 1.0f);
    public bool setPanelScaleOnAttach = true;
    public bool logPoseFollow = false;
    public bool enableViewerFallbackWhenPoseMissing = false;
    public Vector3 viewerFallbackOffset = new Vector3(-0.18f, -0.12f, 0.55f);

    [Header("Robot Link Anchor")]
    public bool anchorToLink67 = true;
    public string link6Name = "link6";
    public string link7Name = "link7";
    public float link67MidBlend = 0.65f;
    public bool link67UseLink7Rotation = true;
    public float link67WorldLift = 0.055f;
    public float link67OutwardFromFront = 0.00f;

    [Header("Diagnostics")]
    public bool logPanelVisibilityDiagnostics = true;
    public bool logPanelVisibilityOnlyWhenHidden = false;
    public float panelVisibilityLogInterval = 2.0f;

    [Header("Visibility Rescue")]
    public bool rescuePanelWhenOutOfView = false;
    public Vector3 rescueViewerOffset = new Vector3(-0.16f, -0.10f, 0.55f);

    [Header("Viewer HUD Fail-Safe")]
    public bool forceViewerHudMode = false;
    public Vector3 viewerHudOffset = new Vector3(-0.12f, -0.06f, 0.45f);
    public Vector3 viewerHudScale = new Vector3(0.22f, 0.13f, 1.0f);
    public bool forceHudAlwaysOnTop = true;
    public int logFirstAppliedFrameStats = 3;

    [Header("Startup Stabilization")]
    public bool forceViewerHudUntilSceneStable = true;
    public int stableChecksBeforeAnchorMount = 3;
    public float minReasonableViewerY = -2.0f;
    public float maxReasonableViewerY = 3.5f;

    private SubscriberSocket _sub;
    private NetMQPoller _poller;
    private volatile bool _shuttingDown;
    private readonly object _netMqLock = new object();
    private readonly object _pendingFrameLock = new object();
    private readonly object _pendingPoseLock = new object();
    private Texture2D[] _rgbTex;
    private PendingTextureFrame[] _pendingFrames = Array.Empty<PendingTextureFrame>();
    private int _logged;
    private Transform _sceneRoot;
    private bool _sceneAttachLogged;
    private long _lastRxTick;
    private float _nextReconnectAt;
    private bool _hasFrontCameraPose;
    private Vector3 _frontCameraPos;
    private bool _hasLatestFollowPose;
    private Vector3 _latestFollowPosePos;
    private Quaternion _latestFollowPoseRot = Quaternion.identity;
    private Renderer _followPanelRenderer;
    private Camera _viewerCamera;
    private int _viewerCameraInstanceId = int.MinValue;
    private string _viewerCameraSource = "unresolved";
    private bool _loggedPoseFallback;
    private bool _loggedFirstPanelPose;
    private bool _loggedFirstAnchorAppliedPose;
    private bool _loggedAnchorWaiting;
    private bool _hadUsableRendererAtStart;
    private Transform _link6Tf;
    private Transform _link7Tf;
    private Transform _panelAnchorTf;
    private bool _linkAnchorLogOnce;
    private bool _panelAnchorLogOnce;
    private bool _panelAnchorUsedFallback;
    private string _panelAnchorResolvedName = string.Empty;
    private bool _panelRescueActive;
    private float _nextPanelVisibilityLogAt;
    private int _appliedFrameStatsLogged;
    private bool _calibratedOffsetFromTarget;
    private string _lastPoseSource = "startup";
    private QuestTrackingOriginGuard _trackingOriginGuard;
    private int _stableHudChecks;
    private bool _anchorMountUnlocked;
    private bool _loggedStartupHudHold;
    private float _lastObservedHudRecoveryAt = float.NegativeInfinity;
    private static readonly double TimestampToSeconds = 1.0 / Stopwatch.Frequency;
    private bool _hasPendingFrontCameraPose;
    private Vector3 _pendingFrontCameraPos;
    private bool _hasPendingFollowPose;
    private Vector3 _pendingFollowPosePos;
    private Quaternion _pendingFollowPoseRot = Quaternion.identity;
    private long _rxCount;
    private float _nextHeartbeatAt;

    private sealed class PendingTextureFrame
    {
        public string CameraName;
        public byte[] JpegBytes;
        public string SourceLabel;
    }

    void Start()
    {
        AsyncIO.ForceDotNet.Force();
        _hadUsableRendererAtStart = HasUsableRenderer();
        WarnOnSceneWiringAmbiguity();
        if (forceFollowCameraSettingsOnStart)
        {
            followCameraPose = true;
            if (string.IsNullOrWhiteSpace(followCameraName))
                followCameraName = "wrist";
            strictCameraFilter = false;
            singleRendererShowsFollowCameraOnly = true;
        }
        if (string.IsNullOrWhiteSpace(followCameraName))
            followCameraName = "wrist";

        if (cameras == null || cameras.Length == 0)
            cameras = new[] { "wrist" };
        cameras = EnsureCameraListed(cameras, followCameraName);

        if (resetLegacyRendererBindingsOnStart)
            targetRenderers = Array.Empty<Renderer>();

        if (forceDedicatedFollowPanel && !HasUsableRenderer())
            EnsureDedicatedFollowPanel();

        if (!HasUsableRenderer())
            autoCreatePanelIfMissing = true;

        // Prefer a dedicated wrist panel if no valid renderer is assigned.
        if (!HasUsableRenderer() && autoCreatePanelIfMissing)
        {
            var autoPanelRenderer = CreateAutoPanelRenderer();
            if (autoPanelRenderer != null)
            {
                _followPanelRenderer = autoPanelRenderer;
                targetRenderers = new[] { autoPanelRenderer };
                followRendererIndex = 0;
                followCameraPose = true;
                strictCameraFilter = false;
                Debug.Log("[SimPubRgbdSubscriber] Created runtime wrist panel renderer.");
            }
        }

        // Fallback: use Renderer on same GO only if auto-panel was disabled/failed.
        if (!HasUsableRenderer())
        {
            var r = GetComponent<Renderer>();
            if (r != null)
            {
                targetRenderers = new[] { r };
                Debug.LogWarning("[SimPubRgbdSubscriber] Falling back to Renderer on same GameObject.");
            }
        }

        if (TryResolveRenderer(followRendererIndex, out var preflightRenderer, out _))
        {
            if (avoidPointCloudRendererAsRgbdTarget && RendererHasPointCloudComponents(preflightRenderer))
            {
                Debug.LogWarning(
                    "[SimPubRgbdSubscriber] Current renderer is shared with point-cloud components. "
                    + "Creating a dedicated RGBD panel to avoid renderer/material conflicts."
                );
                ReplaceWithDedicatedPanel();
            }
        }

        if (!HasUsableRenderer())
        {
            Debug.LogError("[SimPubRgbdSubscriber] targetRenderers is empty. Assign a Quad Renderer (or put this script on the Quad).");
            enabled = false;
            return;
        }
        if (TryResolveRenderer(followRendererIndex, out var activeRenderer, out var activeRendererIdx))
        {
            EnsureRendererTextureCompatibility(activeRenderer);
            Debug.Log(
                $"[SimPubRgbdSubscriber] Active panel renderer='{activeRenderer.name}' go='{activeRenderer.gameObject.name}' "
                + $"idx={activeRendererIdx} dedicated={ReferenceEquals(activeRenderer, _followPanelRenderer)} subscriberId={GetInstanceID()}"
            );
            if (!_hadUsableRendererAtStart && ReferenceEquals(activeRenderer, _followPanelRenderer))
            {
                Debug.LogWarning(
                    $"[SimPubRgbdSubscriber] No valid target renderer was assigned at startup. "
                    + $"Using runtime-created panel '{activeRenderer.gameObject.name}' on subscriberId={GetInstanceID()}."
                );
            }
        }

        ResolveViewerCamera(logSelection: true);
        if (calibrateOffsetFromTargetAtRuntime)
        {
            Debug.LogWarning(
                $"[SimPubRgbdSubscriber] Runtime panel calibration is enabled on subscriberId={GetInstanceID()}. "
                + "Manual panel offset tuning will be overridden after the first pose application."
            );
        }

        EnsureTextureCacheSize();

        _lastRxTick = Stopwatch.GetTimestamp();
        StartNetMq();
        Debug.Log(
            $"[SimPubRgbdSubscriber] Runtime config subscriberId={GetInstanceID()} follow='{followCameraName}' "
            + $"forcePanel={forceDedicatedFollowPanel} parentToScene={parentRendererToSimScene} "
            + $"offset={panelOffsetLocal} offsetWorldAxes={panelOffsetInWorldAxes} viewerTilt={viewerFacingEulerOffset} flattenViewerFacing={flattenViewerFacing} "
            + $"linkLift={link67WorldLift:F3} linkOutward={link67OutwardFromFront:F3} calibrate={calibrateOffsetFromTargetAtRuntime}"
        );
    }

    void Update()
    {
        ApplyPendingPoseUpdates();
        ApplyPendingTextureFrames();

        if (followCameraPose && parentRendererToSimScene && _sceneRoot == null)
            TryResolveSceneRoot();

        // Re-apply placement every frame so the panel snaps into place as soon as
        // scene/link anchors resolve, even if wrist frames arrive before scene spawn.
        if (followCameraPose)
        {
            if (TryResolveRenderer(followRendererIndex, out var followRenderer, out _))
            {
                ApplyPoseFollow(followRenderer.transform, _hasLatestFollowPose, _latestFollowPosePos, _latestFollowPoseRot);
                LogPanelVisibilityDiagnostics(followRenderer);
            }
        }

        if (autoReconnectOnIdle && Time.unscaledTime >= _nextReconnectAt)
        {
            double idleFor =
                (Stopwatch.GetTimestamp() - Interlocked.Read(ref _lastRxTick))
                * TimestampToSeconds;
            if (idleFor >= Mathf.Max(0.5f, reconnectIdleSeconds))
            {
                Debug.LogWarning($"[SimPubRgbdSubscriber] Idle for {idleFor:F2}s. Reconnecting socket...");
                ReconnectNetMq();
            }
        }

        if (heartbeatLogIntervalSeconds > 0f && Time.unscaledTime >= _nextHeartbeatAt)
        {
            _nextHeartbeatAt = Time.unscaledTime + heartbeatLogIntervalSeconds;
            double idleAge = (Stopwatch.GetTimestamp() - Interlocked.Read(ref _lastRxTick)) * TimestampToSeconds;
            Vector3 panelPos = Vector3.zero;
            string panelName = "<none>";
            if (TryResolveRenderer(followRendererIndex, out var rendererForLog, out _) && rendererForLog != null)
            {
                panelPos = rendererForLog.transform.position;
                panelName = rendererForLog.gameObject.name;
            }
            Debug.Log(
                $"[SimPubRgbdSubscriber] hb rx={Interlocked.Read(ref _rxCount)} "
                + $"idle_s={idleAge:F2} mode={textureSourceMode} "
                + $"hasFollowPose={_hasLatestFollowPose} panel='{panelName}' panelWorld={panelPos} "
                + $"followCameraPose={followCameraPose} viewerFallback={enableViewerFallbackWhenPoseMissing} "
                + $"endpoint=tcp://{publisherIp}:{topicPort}"
            );
        }
    }

    void OnDisable() => Shutdown();
    void OnDestroy() => Shutdown();

    private void Shutdown()
    {
        if (_shuttingDown) return;
        _shuttingDown = true;

        SubscriberSocket sub = null;
        NetMQPoller poller = null;
        lock (_netMqLock)
        {
            sub = _sub;
            poller = _poller;
            _sub = null;
            _poller = null;
        }

        try { if (sub != null) sub.ReceiveReady -= OnMsg; } catch { }
        // Stop() blocks until the poll loop exits so the socket isn't disposed out from under
        // a running poller (the "Must not be disposed" crash on scene exit/stop).
        try { if (poller != null && poller.IsRunning) poller.Stop(); } catch { }
        try { sub?.Close(); } catch { }
        try { sub?.Dispose(); } catch { }
        try { poller?.Dispose(); } catch { }

        // IMPORTANT: Do not call NetMQConfig.Cleanup() here.
    }

    private void StartNetMq()
    {
        var sub = new SubscriberSocket();
        sub.Options.ReceiveHighWatermark = 2;
        sub.Options.Linger = TimeSpan.Zero;
        sub.Connect($"tcp://{publisherIp}:{topicPort}");
        var subscribedTopics = SubscribeConfiguredTopics(sub);

        sub.ReceiveReady += OnMsg;
        var poller = new NetMQPoller { sub };
        lock (_netMqLock)
        {
            _sub = sub;
            _poller = poller;
        }
        poller.RunAsync();

        Interlocked.Exchange(ref _lastRxTick, Stopwatch.GetTimestamp());
        _nextReconnectAt = Time.unscaledTime + Mathf.Max(0.5f, reconnectCooldownSeconds);
        Debug.Log(
            $"[SimPubRgbdSubscriber] Connected tcp://{publisherIp}:{topicPort} "
            + $"textureSource={textureSourceMode} subscriptions=[{string.Join(", ", subscribedTopics)}]"
        );
    }

    /// <summary>Re-point this subscriber at a different session and reconnect.
    ///
    /// Combined view switches sessions in place (no scene reload). Without this the
    /// subscriber kept its socket on the PREVIOUSLY selected session, which inflated that
    /// session's ZMQ peer count and kept Python holding it in ACTIVE (full point-cloud)
    /// mode after it was deselected.</summary>
    public void Reconfigure(string ip, int newTopicPort)
    {
        if (publisherIp == ip && topicPort == newTopicPort)
            return;
        publisherIp = ip;
        topicPort = newTopicPort;
        ReconnectNetMq();
        Debug.Log($"[SimPubRgbdSubscriber] reconfigured → tcp://{publisherIp}:{topicPort}");
    }

    private void ReconnectNetMq()
    {
        Shutdown();
        _shuttingDown = false;
        StartNetMq();
    }

    private void OnMsg(object sender, NetMQSocketEventArgs e)
    {
        if (_shuttingDown) return;

        string topic;
        try { topic = e.Socket.ReceiveFrameString(); }
        catch { return; }  // also swallows TerminatingException if context dies on the first frame

        byte[] blob;
        bool gotBlob;
        try { gotBlob = e.Socket.TryReceiveFrameBytes(out blob); }
        catch (TerminatingException) { _shuttingDown = true; try { e.Socket.ReceiveReady -= OnMsg; } catch { } return; }
        catch (ObjectDisposedException) { _shuttingDown = true; try { e.Socket.ReceiveReady -= OnMsg; } catch { } return; }
        if (!gotBlob || blob == null)
            return;

        if (textureSourceMode == WristPanelTextureSourceMode.StandaloneRgb)
        {
            HandleStandaloneRgbMessage(topic, blob);
            return;
        }

        HandleRgbdMessage(topic, blob);
    }

    private void HandleStandaloneRgbMessage(string topic, byte[] blob)
    {
        string expectedTopic = ResolveStandaloneRgbTopic();
        if (!string.Equals(topic, expectedTopic, StringComparison.Ordinal))
            return;

        Interlocked.Exchange(ref _lastRxTick, Stopwatch.GetTimestamp());
        LogReceivedFrame(topic, blob.Length);

        int rendererIdx = ResolvePendingRendererIndex(followCameraName, isFollowCam: true);
        StorePendingTextureFrame(rendererIdx, new PendingTextureFrame
        {
            CameraName = string.IsNullOrWhiteSpace(followCameraName) ? "wrist" : followCameraName,
            JpegBytes = CopyBytes(blob),
            SourceLabel = "rgb",
        });
    }

    private void HandleRgbdMessage(string topic, byte[] blob)
    {
        if (!topic.StartsWith(topicPrefix, StringComparison.Ordinal) || !topic.EndsWith("/rgbd", StringComparison.Ordinal))
            return;

        Interlocked.Exchange(ref _lastRxTick, Stopwatch.GetTimestamp());
        LogReceivedFrame(topic, blob.Length);

        try
        {
            if (blob.Length < 4) return;

            int jsonLen = BitConverter.ToInt32(blob, 0);
            if (!BitConverter.IsLittleEndian)
            {
                var tmp = new byte[4];
                Buffer.BlockCopy(blob, 0, tmp, 0, 4);
                Array.Reverse(tmp);
                jsonLen = BitConverter.ToInt32(tmp, 0);
            }

            if (jsonLen <= 0 || jsonLen > blob.Length - 4) return;

            string json = Encoding.UTF8.GetString(blob, 4, jsonLen);
            var hdr = JsonUtility.FromJson<RgbdHeader>(json);
            if (hdr == null || string.IsNullOrEmpty(hdr.cam_name)) return;

            int offset = 4 + jsonLen;
            if (hdr.rgb_len <= 0 || offset + hdr.rgb_len > blob.Length) return;

            var rgbJpg = new byte[hdr.rgb_len];
            Buffer.BlockCopy(blob, offset, rgbJpg, 0, hdr.rgb_len);

            bool isFollowCam = followCameraPose
                && string.Equals(hdr.cam_name, followCameraName, StringComparison.OrdinalIgnoreCase);
            bool singleRendererMode = targetRenderers == null || targetRenderers.Length <= 1;

            Vector3 camPosePos = Vector3.zero;
            Quaternion camPoseRot = Quaternion.identity;
            bool hasCamPose = TryExtractCamPose(hdr, out camPosePos, out camPoseRot);
            bool isFrontCam = string.Equals(hdr.cam_name, frontCameraName, StringComparison.OrdinalIgnoreCase);

            if (isFrontCam && hasCamPose)
                StorePendingFrontCameraPose(camPosePos);

            if (isFollowCam && hasCamPose)
                StorePendingFollowPose(camPosePos, camPoseRot);

            if (singleRendererShowsFollowCameraOnly && singleRendererMode && followCameraPose && !isFollowCam)
                return;

            int camIdx = Array.IndexOf(cameras, hdr.cam_name);
            if (strictCameraFilter && camIdx < 0 && !isFollowCam)
                return;

            int rendererIdx = ResolvePendingRendererIndex(hdr.cam_name, isFollowCam);
            StorePendingTextureFrame(rendererIdx, new PendingTextureFrame
            {
                CameraName = hdr.cam_name,
                JpegBytes = rgbJpg,
                SourceLabel = "rgbd",
            });
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"[SimPubRgbdSubscriber] decode failed: {ex.Message}");
        }
    }

    private string[] EnsureCameraListed(string[] list, string camName)
    {
        if (string.IsNullOrWhiteSpace(camName))
            return list ?? Array.Empty<string>();
        if (list == null || list.Length == 0)
            return new[] { camName };

        foreach (var c in list)
        {
            if (string.Equals(c, camName, StringComparison.OrdinalIgnoreCase))
                return list;
        }

        var outList = new string[list.Length + 1];
        outList[0] = camName;
        Array.Copy(list, 0, outList, 1, list.Length);
        return outList;
    }

    private void EnsureDedicatedFollowPanel()
    {
        if (!forceDedicatedFollowPanel)
            return;

        if (_followPanelRenderer != null)
            return;

        var renderer = CreateAutoPanelRenderer();
        if (renderer == null)
            return;

        _followPanelRenderer = renderer;
        targetRenderers = new[] { renderer };
        followRendererIndex = 0;
        EnsureTextureCacheSize();
        Debug.Log("[SimPubRgbdSubscriber] Dedicated runtime wrist panel is active.");
    }

    private Renderer CreateAutoPanelRenderer()
    {
        GameObject panel = GameObject.CreatePrimitive(PrimitiveType.Quad);
        panel.name = autoPanelName;
        panel.transform.SetParent(transform, worldPositionStays: false);
        panel.transform.localPosition = Vector3.zero;
        panel.transform.localRotation = Quaternion.identity;
        panel.transform.localScale = panelScale;

        var collider = panel.GetComponent<Collider>();
        if (collider != null)
            Destroy(collider);

        var renderer = panel.GetComponent<Renderer>();
        if (renderer == null)
            return null;

        renderer.shadowCastingMode = ShadowCastingMode.Off;
        renderer.receiveShadows = false;
        renderer.allowOcclusionWhenDynamic = false;

        if (autoPanelUseUnlitMaterial)
        {
            Shader shader = Shader.Find("Universal Render Pipeline/Unlit");
            if (shader == null)
                shader = Shader.Find("Unlit/Texture");
            if (shader != null)
                renderer.material = new Material(shader);
        }

        var m = renderer.material;
        if (m != null)
        {
            if (m.HasProperty("_Cull"))
                m.SetInt("_Cull", (int)CullMode.Off);
            if (m.HasProperty("_BaseColor"))
                m.SetColor("_BaseColor", Color.white);
            if (m.HasProperty("_Color"))
                m.SetColor("_Color", Color.white);
        }

        if (autoPanelFlipY)
        {
            var mat = renderer.material;
            if (mat != null)
            {
                Vector2 flip = new Vector2(1f, -1f);
                if (mat.HasProperty("_MainTex"))
                    mat.mainTextureScale = flip;
                if (mat.HasProperty("_BaseMap"))
                    mat.SetTextureScale("_BaseMap", flip);
            }
        }

        return renderer;
    }

    private bool HasUsableRenderer()
    {
        if (targetRenderers == null || targetRenderers.Length == 0)
            return false;
        for (int i = 0; i < targetRenderers.Length; i++)
        {
            if (targetRenderers[i] != null)
                return true;
        }
        return false;
    }

    private void EnsureTextureCacheSize()
    {
        int n = (targetRenderers == null) ? 0 : targetRenderers.Length;
        if (n <= 0)
        {
            _rgbTex = Array.Empty<Texture2D>();
            _pendingFrames = Array.Empty<PendingTextureFrame>();
            return;
        }

        if (_rgbTex == null || _rgbTex.Length != n)
            _rgbTex = new Texture2D[n];
        if (_pendingFrames == null || _pendingFrames.Length != n)
            _pendingFrames = new PendingTextureFrame[n];
    }

    private void LogReceivedFrame(string topic, int byteCount)
    {
        Interlocked.Increment(ref _rxCount);
        if (_logged < logFirstMessages)
        {
            Debug.Log($"[SimPubRgbdSubscriber] RX {topic} bytes={byteCount}");
            _logged++;
        }
    }

    private string ResolveStandaloneRgbTopic()
    {
        if (!string.IsNullOrWhiteSpace(standaloneRgbTopic))
            return standaloneRgbTopic;

        string camName = string.IsNullOrWhiteSpace(followCameraName) ? "wrist" : followCameraName;
        return $"{topicPrefix}/{camName}/rgb";
    }

    private string ResolveRgbdTopic(string cameraName)
    {
        if (string.IsNullOrWhiteSpace(cameraName))
            cameraName = string.IsNullOrWhiteSpace(followCameraName) ? "wrist" : followCameraName;
        return $"{topicPrefix}/{cameraName}/rgbd";
    }

    private string[] SubscribeConfiguredTopics(SubscriberSocket sub)
    {
        var topics = BuildSubscribedTopics();
        if (topics.Length == 0)
        {
            sub.Subscribe("");
            return new[] { "<all>" };
        }

        for (int i = 0; i < topics.Length; i++)
            sub.Subscribe(topics[i]);

        return topics;
    }

    private string[] BuildSubscribedTopics()
    {
        if (textureSourceMode == WristPanelTextureSourceMode.StandaloneRgb)
            return new[] { ResolveStandaloneRgbTopic() };

        var topics = new List<string>();
        bool singleRendererMode = targetRenderers == null || targetRenderers.Length <= 1;

        if (singleRendererShowsFollowCameraOnly && singleRendererMode && followCameraPose)
        {
            AddUniqueTopic(topics, ResolveRgbdTopic(followCameraName));
            if (orientTowardFrontCamera)
                AddUniqueTopic(topics, ResolveRgbdTopic(frontCameraName));
            return topics.ToArray();
        }

        if (cameras != null)
        {
            for (int i = 0; i < cameras.Length; i++)
                AddUniqueTopic(topics, ResolveRgbdTopic(cameras[i]));
        }

        if (followCameraPose)
            AddUniqueTopic(topics, ResolveRgbdTopic(followCameraName));
        if (orientTowardFrontCamera)
            AddUniqueTopic(topics, ResolveRgbdTopic(frontCameraName));

        return topics.ToArray();
    }

    private static void AddUniqueTopic(List<string> topics, string topic)
    {
        if (string.IsNullOrWhiteSpace(topic))
            return;

        for (int i = 0; i < topics.Count; i++)
        {
            if (string.Equals(topics[i], topic, StringComparison.Ordinal))
                return;
        }

        topics.Add(topic);
    }

    private int ResolvePendingRendererIndex(string cameraName, bool isFollowCam)
    {
        int rendererCount = (_rgbTex != null && _rgbTex.Length > 0)
            ? _rgbTex.Length
            : ((targetRenderers != null && targetRenderers.Length > 0) ? targetRenderers.Length : 1);
        int camIdx = Array.IndexOf(cameras, cameraName);
        int rendererIdx = isFollowCam ? followRendererIndex : (camIdx < 0 ? 0 : camIdx);
        return Mathf.Clamp(rendererIdx, 0, Mathf.Max(0, rendererCount - 1));
    }

    private static byte[] CopyBytes(byte[] payload)
    {
        if (payload == null || payload.Length == 0)
            return Array.Empty<byte>();

        var copy = new byte[payload.Length];
        Buffer.BlockCopy(payload, 0, copy, 0, payload.Length);
        return copy;
    }

    private void StorePendingTextureFrame(int rendererIdx, PendingTextureFrame frame)
    {
        if (frame == null)
            return;

        lock (_pendingFrameLock)
        {
            if (_pendingFrames == null || _pendingFrames.Length == 0)
                EnsureTextureCacheSize();
            if (_pendingFrames == null || _pendingFrames.Length == 0)
                return;

            rendererIdx = Mathf.Clamp(rendererIdx, 0, _pendingFrames.Length - 1);
            _pendingFrames[rendererIdx] = frame;
        }
    }

    private void StorePendingFrontCameraPose(Vector3 pos)
    {
        lock (_pendingPoseLock)
        {
            _pendingFrontCameraPos = pos;
            _hasPendingFrontCameraPose = true;
        }
    }

    private void StorePendingFollowPose(Vector3 pos, Quaternion rot)
    {
        lock (_pendingPoseLock)
        {
            _pendingFollowPosePos = pos;
            _pendingFollowPoseRot = rot;
            _hasPendingFollowPose = true;
        }
    }

    private void ApplyPendingPoseUpdates()
    {
        lock (_pendingPoseLock)
        {
            if (_hasPendingFrontCameraPose)
            {
                _frontCameraPos = _pendingFrontCameraPos;
                _hasFrontCameraPose = true;
                _hasPendingFrontCameraPose = false;
            }

            if (_hasPendingFollowPose)
            {
                _latestFollowPosePos = _pendingFollowPosePos;
                _latestFollowPoseRot = _pendingFollowPoseRot;
                _hasLatestFollowPose = true;
                _hasPendingFollowPose = false;
            }
        }
    }

    private void ApplyPendingTextureFrames()
    {
        PendingTextureFrame[] framesToApply = null;
        lock (_pendingFrameLock)
        {
            if (_pendingFrames == null || _pendingFrames.Length == 0)
                return;

            bool anyPending = false;
            framesToApply = new PendingTextureFrame[_pendingFrames.Length];
            for (int i = 0; i < _pendingFrames.Length; i++)
            {
                framesToApply[i] = _pendingFrames[i];
                if (framesToApply[i] != null)
                {
                    anyPending = true;
                    _pendingFrames[i] = null;
                }
            }

            if (!anyPending)
                return;
        }

        for (int i = 0; i < framesToApply.Length; i++)
        {
            var frame = framesToApply[i];
            if (frame == null || frame.JpegBytes == null || frame.JpegBytes.Length == 0)
                continue;

            if (!TryResolveRenderer(i, out var renderer, out var resolvedRendererIdx))
                continue;

            int texIdx = resolvedRendererIdx >= 0 ? resolvedRendererIdx : i;
            if (_rgbTex[texIdx] == null)
                _rgbTex[texIdx] = new Texture2D(2, 2, TextureFormat.RGB24, false);

            bool loaded = _rgbTex[texIdx].LoadImage(frame.JpegBytes, markNonReadable: false);
            if (!loaded)
            {
                Debug.LogWarning(
                    $"[SimPubRgbdSubscriber] LoadImage failed for cam='{frame.CameraName}' "
                    + $"source='{frame.SourceLabel}' bytes={frame.JpegBytes.Length}"
                );
                continue;
            }

            var mat = EnforcePanelRenderState(renderer, _rgbTex[texIdx]);
            if (mat == null)
            {
                Debug.LogWarning(
                    $"[SimPubRgbdSubscriber] Renderer material missing for cam='{frame.CameraName}' "
                    + $"on '{renderer.gameObject.name}'."
                );
                continue;
            }

            if (!mat.HasProperty("_MainTex") && !mat.HasProperty("_BaseMap"))
            {
                Debug.LogWarning(
                    $"[SimPubRgbdSubscriber] Active shader '{mat.shader.name}' has no _MainTex/_BaseMap "
                    + $"for cam='{frame.CameraName}'."
                );
                continue;
            }

            if (_appliedFrameStatsLogged < Mathf.Max(0, logFirstAppliedFrameStats))
            {
                _appliedFrameStatsLogged++;
                int cx = Mathf.Clamp(_rgbTex[texIdx].width / 2, 0, _rgbTex[texIdx].width - 1);
                int cy = Mathf.Clamp(_rgbTex[texIdx].height / 2, 0, _rgbTex[texIdx].height - 1);
                Color c = _rgbTex[texIdx].GetPixel(cx, cy);
                Debug.Log(
                    $"[SimPubRgbdSubscriber] Applied RGB frame cam='{frame.CameraName}' source='{frame.SourceLabel}' "
                    + $"tex={_rgbTex[texIdx].width}x{_rgbTex[texIdx].height} "
                    + $"centerRGB=({c.r:F2},{c.g:F2},{c.b:F2}) shader='{mat.shader.name}'"
                );
            }
        }
    }

    private bool TryResolveRenderer(int preferredIndex, out Renderer resolved, out int resolvedIndex)
    {
        resolved = null;
        resolvedIndex = -1;

        if (forceDedicatedFollowPanel && !HasUsableRenderer())
        {
            if (_followPanelRenderer == null)
                EnsureDedicatedFollowPanel();

            if (_followPanelRenderer != null)
            {
                targetRenderers = new[] { _followPanelRenderer };
                EnsureTextureCacheSize();
                resolved = _followPanelRenderer;
                resolvedIndex = 0;
                return true;
            }
        }

        if (targetRenderers == null || targetRenderers.Length == 0)
        {
            if (!autoCreatePanelIfMissing)
                return false;

            var autoRenderer = CreateAutoPanelRenderer();
            if (autoRenderer == null)
                return false;

            _followPanelRenderer = autoRenderer;
            targetRenderers = new[] { autoRenderer };
            followRendererIndex = 0;
            EnsureTextureCacheSize();
        }

        if (preferredIndex >= 0 && preferredIndex < targetRenderers.Length && targetRenderers[preferredIndex] != null)
        {
            resolved = targetRenderers[preferredIndex];
            resolvedIndex = preferredIndex;
            EnsureTextureCacheSize();
            return true;
        }

        for (int i = 0; i < targetRenderers.Length; i++)
        {
            if (targetRenderers[i] != null)
            {
                resolved = targetRenderers[i];
                resolvedIndex = i;
                EnsureTextureCacheSize();
                return true;
            }
        }

        if (!autoCreatePanelIfMissing)
            return false;

        var fallbackRenderer = CreateAutoPanelRenderer();
        if (fallbackRenderer == null)
            return false;

        _followPanelRenderer = fallbackRenderer;
        targetRenderers = new[] { fallbackRenderer };
        followRendererIndex = 0;
        EnsureTextureCacheSize();
        resolved = fallbackRenderer;
        resolvedIndex = 0;
        Debug.Log("[SimPubRgbdSubscriber] Created runtime wrist panel renderer (fallback).");
        return true;
    }

    private static bool RendererHasPointCloudComponents(Renderer renderer)
    {
        if (renderer == null)
            return false;

        return renderer.GetComponent<SimplePointCloudRenderer>() != null
            || renderer.GetComponent<SimPubPointCloudSubscriber>() != null
            || renderer.GetComponent<GpuInstancedPointCloudRenderer>() != null
            || renderer.GetComponent<GpuPointCloudSubscriber>() != null;
    }

    private void ReplaceWithDedicatedPanel()
    {
        var renderer = CreateAutoPanelRenderer();
        if (renderer == null)
            return;

        _followPanelRenderer = renderer;
        targetRenderers = new[] { renderer };
        followRendererIndex = 0;
        EnsureTextureCacheSize();
    }

    private void EnsureRendererTextureCompatibility(Renderer renderer)
    {
        if (!forceTextureCompatibleMaterial || renderer == null)
            return;

        var mat = renderer.material;
        if (mat == null)
            return;

        if (mat.HasProperty("_MainTex") || mat.HasProperty("_BaseMap"))
            return;

        Shader shader = Shader.Find("Universal Render Pipeline/Unlit");
        if (shader == null)
            shader = Shader.Find("Unlit/Texture");
        if (shader == null)
            return;

        renderer.material = new Material(shader);
        var m = renderer.material;
        if (m.HasProperty("_Cull"))
            m.SetInt("_Cull", (int)CullMode.Off);
        if (m.HasProperty("_BaseColor"))
            m.SetColor("_BaseColor", Color.white);
        if (m.HasProperty("_Color"))
            m.SetColor("_Color", Color.white);

        Debug.LogWarning(
            $"[SimPubRgbdSubscriber] Replaced incompatible panel material on '{renderer.gameObject.name}' "
            + $"with '{shader.name}' to ensure RGB texture rendering."
        );
    }

    private static void EnsureAlwaysOnTopMaterialState(Renderer renderer)
    {
        if (renderer == null)
            return;
        var mat = renderer.material;
        if (mat == null)
            return;

        if (mat.HasProperty("_ZWrite"))
            mat.SetInt("_ZWrite", 0);
        if (mat.HasProperty("_ZTest"))
            mat.SetInt("_ZTest", (int)CompareFunction.Always);
        if (mat.HasProperty("_Cull"))
            mat.SetInt("_Cull", (int)CullMode.Off);
        mat.renderQueue = (int)RenderQueue.Overlay;
    }

    private Material EnforcePanelRenderState(Renderer renderer, Texture texture)
    {
        if (renderer == null)
            return null;

        renderer.enabled = true;
        renderer.shadowCastingMode = ShadowCastingMode.Off;
        renderer.receiveShadows = false;
        renderer.allowOcclusionWhenDynamic = false;

        EnsureRendererTextureCompatibility(renderer);
        var mat = renderer.material;
        if (mat == null)
            return null;

        if (mat.HasProperty("_Cull"))
            mat.SetInt("_Cull", (int)CullMode.Off);
        if (mat.HasProperty("_BaseColor"))
            mat.SetColor("_BaseColor", Color.white);
        if (mat.HasProperty("_Color"))
            mat.SetColor("_Color", Color.white);

        if (texture != null)
        {
            if (mat.HasProperty("_MainTex"))
                mat.SetTexture("_MainTex", texture);
            if (mat.HasProperty("_BaseMap"))
                mat.SetTexture("_BaseMap", texture);
            mat.mainTexture = texture;
        }

        if (autoPanelFlipY)
        {
            Vector2 flip = new Vector2(1f, -1f);
            if (mat.HasProperty("_MainTex"))
                mat.mainTextureScale = flip;
            if (mat.HasProperty("_BaseMap"))
                mat.SetTextureScale("_BaseMap", flip);
        }

        if (forceViewerHudMode && forceHudAlwaysOnTop)
            EnsureAlwaysOnTopMaterialState(renderer);

        return mat;
    }

    private void WarnOnSceneWiringAmbiguity()
    {
        var subscribers = UnityEngine.Object.FindObjectsByType<SimPubRgbdSubscriber>(
            FindObjectsInactive.Include,
            FindObjectsSortMode.None
        );
        int enabledSubscribers = 0;
        var enabledSubscriberInfo = new StringBuilder();
        for (int i = 0; i < subscribers.Length; i++)
        {
            var sub = subscribers[i];
            if (sub == null || !sub.enabled)
                continue;

            if (enabledSubscribers > 0)
                enabledSubscriberInfo.Append(", ");
            enabledSubscriberInfo.Append(sub.name).Append("#").Append(sub.GetInstanceID());
            enabledSubscribers++;
        }

        if (enabledSubscribers > 1)
        {
            Debug.LogWarning(
                $"[SimPubRgbdSubscriber] Multiple enabled subscribers detected ({enabledSubscribers}): {enabledSubscriberInfo}."
            );
        }

        if (!warnOnMultipleMainCameras)
            return;

        var cameras = UnityEngine.Object.FindObjectsByType<Camera>(
            FindObjectsInactive.Include,
            FindObjectsSortMode.None
        );
        int mainCameraCount = 0;
        var mainCameraInfo = new StringBuilder();
        for (int i = 0; i < cameras.Length; i++)
        {
            var cam = cameras[i];
            if (cam == null || !cam.CompareTag("MainCamera"))
                continue;

            if (mainCameraCount > 0)
                mainCameraInfo.Append(", ");
            mainCameraInfo.Append(cam.name)
                .Append("#").Append(cam.GetInstanceID())
                .Append("(active=").Append(cam.gameObject.activeInHierarchy)
                .Append(",enabled=").Append(cam.enabled)
                .Append(")");
            mainCameraCount++;
        }

        if (mainCameraCount > 1)
        {
            Debug.LogWarning(
                $"[SimPubRgbdSubscriber] Multiple MainCamera-tagged cameras detected ({mainCameraCount}): {mainCameraInfo}."
            );
        }
    }

    private static Camera FindCameraByName(string cameraName)
    {
        if (string.IsNullOrWhiteSpace(cameraName))
            return null;

        var cameras = UnityEngine.Object.FindObjectsByType<Camera>(
            FindObjectsInactive.Include,
            FindObjectsSortMode.None
        );
        for (int i = 0; i < cameras.Length; i++)
        {
            var cam = cameras[i];
            if (cam == null || !string.Equals(cam.name, cameraName, StringComparison.Ordinal))
                continue;

            if (cam.gameObject.activeInHierarchy && cam.enabled)
                return cam;
        }

        return null;
    }

    private static Camera TryGetCameraFromTransform(Transform target)
    {
        if (target == null)
            return null;

        var direct = target.GetComponent<Camera>();
        if (direct != null)
            return direct;

        return target.GetComponentInChildren<Camera>(true);
    }

    private static string GetTransformPath(Transform target)
    {
        if (target == null)
            return "<null>";

        var names = new System.Collections.Generic.List<string>();
        var current = target;
        while (current != null)
        {
            names.Add(current.name);
            current = current.parent;
        }
        names.Reverse();
        return string.Join("/", names);
    }

    private Camera ResolveViewerCamera(bool logSelection = false)
    {
        if (!logSelection && _viewerCamera != null && _viewerCamera.gameObject.activeInHierarchy && _viewerCamera.enabled)
            return _viewerCamera;

        Camera cam = FindCameraByName(preferredViewerCameraName);
        string source = cam != null ? $"preferred-name:{preferredViewerCameraName}" : null;

        if (cam == null)
        {
            cam = TryGetCameraFromTransform(viewerCameraOverride);
            if (cam != null)
                source = "override";
        }

        if (cam == null)
        {
            cam = Camera.main;
            if (cam != null)
                source = "Camera.main";
        }

        if (cam == null)
        {
            var cameras = UnityEngine.Object.FindObjectsByType<Camera>(
                FindObjectsInactive.Include,
                FindObjectsSortMode.None
            );
            for (int i = 0; i < cameras.Length; i++)
            {
                if (cameras[i] == null)
                    continue;
                cam = cameras[i];
                source = "first-camera";
                if (cam.gameObject.activeInHierarchy && cam.enabled)
                    break;
            }
        }

        if (cam == null)
        {
            if (logSelection)
                Debug.LogWarning("[SimPubRgbdSubscriber] Viewer camera resolution failed. No camera is available.");
            _viewerCamera = null;
            _viewerCameraSource = "unresolved";
            _viewerCameraInstanceId = int.MinValue;
            return null;
        }

        bool changed = _viewerCamera != cam || _viewerCameraInstanceId != cam.GetInstanceID() || _viewerCameraSource != source;
        _viewerCamera = cam;
        _viewerCameraSource = source ?? "unknown";
        _viewerCameraInstanceId = cam.GetInstanceID();

        if (logSelection || changed)
        {
            Debug.Log(
                $"[SimPubRgbdSubscriber] Viewer camera resolved source='{_viewerCameraSource}' name='{cam.name}' "
                + $"path='{GetTransformPath(cam.transform)}' id={cam.GetInstanceID()} "
                + $"active={cam.gameObject.activeInHierarchy} enabled={cam.enabled}"
            );
        }

        return _viewerCamera;
    }

    private QuestTrackingOriginGuard ResolveTrackingOriginGuard()
    {
        if (_trackingOriginGuard == null)
            _trackingOriginGuard = QuestTrackingOriginGuard.ResolveSharedGuard();
        return _trackingOriginGuard;
    }

    private bool ApplyViewerHudPose(
        Transform panelTf,
        Renderer panelRenderer,
        Camera viewerCam,
        string poseSource
    )
    {
        if (panelTf == null || viewerCam == null)
            return false;

        if (panelTf.parent != null)
            panelTf.SetParent(null, worldPositionStays: true);

        _lastPoseSource = poseSource;
        Vector3 hudPos =
            viewerCam.transform.position
            + viewerCam.transform.right * viewerHudOffset.x
            + viewerCam.transform.up * viewerHudOffset.y
            + viewerCam.transform.forward * viewerHudOffset.z;
        Vector3 toViewer = viewerCam.transform.position - hudPos;
        Quaternion hudRot = toViewer.sqrMagnitude > 1e-8f
            ? Quaternion.LookRotation(toViewer.normalized, Vector3.up) * Quaternion.Euler(viewerFacingEulerOffset)
            : Quaternion.identity;

        panelTf.position = hudPos;
        panelTf.rotation = hudRot;
        panelTf.localScale = viewerHudScale;

        if (!panelTf.gameObject.activeSelf)
            panelTf.gameObject.SetActive(true);

        if (panelRenderer != null)
        {
            panelRenderer.enabled = true;
            if (forceHudAlwaysOnTop)
                EnsureAlwaysOnTopMaterialState(panelRenderer);
        }

        return true;
    }

    private bool ShouldHoldPanelInViewerHud(Camera viewerCam, out string reason)
    {
        reason = string.Empty;
        if (!forceViewerHudUntilSceneStable)
            return false;

        var trackingOriginGuard = ResolveTrackingOriginGuard();
        if (trackingOriginGuard != null
            && trackingOriginGuard.LastRecoveryAt > _lastObservedHudRecoveryAt)
        {
            _lastObservedHudRecoveryAt = trackingOriginGuard.LastRecoveryAt;
            _anchorMountUnlocked = false;
            _stableHudChecks = 0;
            _loggedStartupHudHold = false;
            _loggedFirstAnchorAppliedPose = false;
            Debug.LogWarning(
                "[SimPubRgbdSubscriber] XR recovery detected; keeping the wrist panel in viewer HUD mode until scene and tracking stabilize again."
            );
        }

        if (_anchorMountUnlocked)
            return false;

        if (_sceneRoot == null)
            TryResolveSceneRoot();

        bool hasSceneRoot = _sceneRoot != null;
        bool runtimeStable = trackingOriginGuard == null || trackingOriginGuard.IsRuntimeStable;
        bool worldFrameStable = trackingOriginGuard == null || trackingOriginGuard.IsWorldFrameStable;
        bool hasViewerCamera = viewerCam != null;
        if (hasSceneRoot && runtimeStable && worldFrameStable && hasViewerCamera)
            _stableHudChecks = Mathf.Min(_stableHudChecks + 1, Mathf.Max(1, stableChecksBeforeAnchorMount));
        else
            _stableHudChecks = 0;

        if (_stableHudChecks >= Mathf.Max(1, stableChecksBeforeAnchorMount))
        {
            _anchorMountUnlocked = true;
            _loggedStartupHudHold = false;
            Debug.Log(
                "[SimPubRgbdSubscriber] Wrist panel leaving viewer HUD stabilization mode. "
                + $"sceneRoot='{_sceneRoot.name}' stableChecks={_stableHudChecks}/{Mathf.Max(1, stableChecksBeforeAnchorMount)} "
                + $"worldFrameStable={worldFrameStable} "
                + $"rigRootDelta={(trackingOriginGuard != null ? trackingOriginGuard.CurrentRigRootWorldDelta.ToString("F3") : "n/a")} "
                + $"trackingSpaceDelta={(trackingOriginGuard != null ? trackingOriginGuard.CurrentTrackingSpaceWorldDelta.ToString("F3") : "n/a")}"
            );
            return false;
        }

        if (!hasSceneRoot)
            reason = "scene-root-pending";
        else if (!hasViewerCamera)
            reason = "viewer-camera-missing";
        else if (!runtimeStable)
            reason = "tracking-unstable";
        else if (!worldFrameStable)
            reason = trackingOriginGuard != null && !trackingOriginGuard.HasWorldFrameBaseline
                ? "world-frame-baseline-pending"
                : $"world-frame-drift rig={trackingOriginGuard.CurrentRigRootWorldDelta.ToString("F3")}";
        else
            reason = "stabilizing";

        if (!_loggedStartupHudHold)
        {
            Debug.LogWarning(
                "[SimPubRgbdSubscriber] Holding wrist panel in viewer HUD mode until startup stabilizes. "
                + $"reason='{reason}' stableChecks={_stableHudChecks}/{Mathf.Max(1, stableChecksBeforeAnchorMount)}"
            );
            _loggedStartupHudHold = true;
        }

        return true;
    }

    private bool TryExtractCamPose(RgbdHeader hdr, out Vector3 pos, out Quaternion rot)
    {
        pos = Vector3.zero;
        rot = Quaternion.identity;
        if (hdr == null || hdr.cam_pose_unity == null)
            return false;

        if (hdr.cam_pose_unity.pos == null || hdr.cam_pose_unity.pos.Length < 3)
            return false;
        if (hdr.cam_pose_unity.quat_xyzw == null || hdr.cam_pose_unity.quat_xyzw.Length < 4)
            return false;

        pos = new Vector3(
            hdr.cam_pose_unity.pos[0],
            hdr.cam_pose_unity.pos[1],
            hdr.cam_pose_unity.pos[2]
        );
        rot = new Quaternion(
            hdr.cam_pose_unity.quat_xyzw[0],
            hdr.cam_pose_unity.quat_xyzw[1],
            hdr.cam_pose_unity.quat_xyzw[2],
            hdr.cam_pose_unity.quat_xyzw[3]
        );
        return true;
    }

    private void TryResolveSceneRoot()
    {
        Transform root = null;
        if (SimSceneSpawner.Instance != null)
            root = SimSceneSpawner.Instance.GetSceneTransform(simSceneName);
        if (root == null)
        {
            var go = GameObject.Find(simSceneName);
            if (go != null) root = go.transform;
        }
        if (root == null) return;

        if (_sceneRoot != root)
        {
            _link6Tf = null;
            _link7Tf = null;
            _panelAnchorTf = null;
            _linkAnchorLogOnce = false;
            _panelAnchorLogOnce = false;
            _panelAnchorUsedFallback = false;
            _panelAnchorResolvedName = string.Empty;
            _loggedFirstAnchorAppliedPose = false;
            _anchorMountUnlocked = false;
            _stableHudChecks = 0;
            _loggedStartupHudHold = false;
        }
        _sceneRoot = root;
        if (!_sceneAttachLogged)
        {
            Debug.Log($"[SimPubRgbdSubscriber] Pose-follow scene root: '{root.name}'");
            _sceneAttachLogged = true;
        }
    }

    private static Transform FindDescendantByName(Transform root, string nodeName)
    {
        if (root == null || string.IsNullOrWhiteSpace(nodeName))
            return null;
        var all = root.GetComponentsInChildren<Transform>(true);
        foreach (var t in all)
        {
            if (string.Equals(t.name, nodeName, StringComparison.Ordinal))
                return t;
        }
        return null;
    }

    private bool TryResolveLinkAnchors()
    {
        if (!anchorToLink67 || _sceneRoot == null)
            return false;

        if (_link6Tf == null)
            _link6Tf = FindDescendantByName(_sceneRoot, link6Name);
        if (_link7Tf == null)
            _link7Tf = FindDescendantByName(_sceneRoot, link7Name);

        bool ok = _link6Tf != null && _link7Tf != null;
        if (ok && !_linkAnchorLogOnce)
        {
            Debug.Log($"[SimPubRgbdSubscriber] Link67 anchor resolved: link6='{_link6Tf.name}' link7='{_link7Tf.name}'");
            _linkAnchorLogOnce = true;
        }
        return ok;
    }

    private bool TryResolvePanelAnchorTransform(out Transform anchorTf)
    {
        anchorTf = null;
        if (!parentPanelUnderAnchorLink)
            return false;

        if (_sceneRoot == null)
            TryResolveSceneRoot();
        if (_sceneRoot == null)
            return false;

        if (_panelAnchorTf != null)
        {
            anchorTf = _panelAnchorTf;
            return true;
        }

        if (!string.IsNullOrWhiteSpace(panelAnchorLinkName))
        {
            var primary = FindDescendantByName(_sceneRoot, panelAnchorLinkName);
            if (primary != null)
            {
                _panelAnchorTf = primary;
                _panelAnchorResolvedName = primary.name;
                _panelAnchorUsedFallback = false;
            }
        }

        if (_panelAnchorTf == null && !string.IsNullOrWhiteSpace(panelAnchorFallbackLinkName))
        {
            var fallback = FindDescendantByName(_sceneRoot, panelAnchorFallbackLinkName);
            if (fallback != null)
            {
                _panelAnchorTf = fallback;
                _panelAnchorResolvedName = fallback.name;
                _panelAnchorUsedFallback = true;
            }
        }

        if (_panelAnchorTf != null && !_panelAnchorLogOnce)
        {
            string source = _panelAnchorUsedFallback ? "fallback" : "primary";
            Debug.Log(
                $"[SimPubRgbdSubscriber] Panel anchor resolved source='{source}' "
                + $"anchor='{_panelAnchorResolvedName}' path='{GetTransformPath(_panelAnchorTf)}'"
            );
            _panelAnchorLogOnce = true;
        }

        anchorTf = _panelAnchorTf;
        return anchorTf != null;
    }

    private void LogPanelVisibilityDiagnostics(Renderer panelRenderer)
    {
        if (!logPanelVisibilityDiagnostics)
            return;
        if (Time.unscaledTime < _nextPanelVisibilityLogAt)
            return;
        _nextPanelVisibilityLogAt = Time.unscaledTime + Mathf.Max(0.25f, panelVisibilityLogInterval);

        if (panelRenderer == null)
        {
            Debug.LogWarning("[SimPubRgbdSubscriber] Panel diagnostics: renderer is null.");
            return;
        }

        Camera cam = ResolveViewerCamera();
        if (cam == null)
        {
            Debug.LogWarning("[SimPubRgbdSubscriber] Panel diagnostics: no active camera.");
            return;
        }

        Vector3 panelCenter = panelRenderer.bounds.center;
        Vector3 vp = cam.WorldToViewportPoint(panelCenter);
        bool inFront = vp.z > 0f;
        bool insideViewport = inFront && vp.x >= 0f && vp.x <= 1f && vp.y >= 0f && vp.y <= 1f;
        bool active = panelRenderer.gameObject.activeInHierarchy;
        bool enabled = panelRenderer.enabled;
        bool visible = panelRenderer.isVisible;

        if (logPanelVisibilityOnlyWhenHidden && active && enabled && visible && insideViewport)
            return;

        var mat = panelRenderer.sharedMaterial != null ? panelRenderer.material : null;
        Texture tex = null;
        string shaderName = "<no-material>";
        if (mat != null)
        {
            shaderName = mat.shader != null ? mat.shader.name : "<no-shader>";
            if (mat.HasProperty("_BaseMap"))
                tex = mat.GetTexture("_BaseMap");
            if (tex == null && mat.HasProperty("_MainTex"))
                tex = mat.GetTexture("_MainTex");
            if (tex == null)
                tex = mat.mainTexture;
        }
        string texInfo = tex == null ? "none" : $"{tex.width}x{tex.height}";
        var trackingOriginGuard = ResolveTrackingOriginGuard();
        bool runtimeStable = trackingOriginGuard == null || trackingOriginGuard.IsRuntimeStable;
        bool worldFrameStable = trackingOriginGuard == null || trackingOriginGuard.IsWorldFrameStable;

        Debug.Log(
            "[SimPubRgbdSubscriber] Panel diagnostics: "
            + $"active={active} enabled={enabled} visible={visible} inFront={inFront} inViewport={insideViewport} "
            + $"vp=({vp.x:F2},{vp.y:F2},{vp.z:F2}) "
            + $"panelPos={panelRenderer.transform.position} panelRot={panelRenderer.transform.rotation.eulerAngles} panelScale={panelRenderer.transform.localScale} "
            + $"camera='{cam.name}' camPath='{GetTransformPath(cam.transform)}' camSource='{_viewerCameraSource}' camPos={cam.transform.position} "
            + $"shader='{shaderName}' tex={texInfo} poseSource='{_lastPoseSource}' "
            + $"anchor='{_panelAnchorResolvedName}' anchorFallback={_panelAnchorUsedFallback} "
            + $"anchorMountUnlocked={_anchorMountUnlocked} runtimeStable={runtimeStable} worldFrameStable={worldFrameStable} "
            + $"hasWorldFrameBaseline={(trackingOriginGuard == null || trackingOriginGuard.HasWorldFrameBaseline)} "
            + $"rigRootDelta={(trackingOriginGuard != null ? trackingOriginGuard.CurrentRigRootWorldDelta.ToString("F3") : "n/a")} "
            + $"trackingSpaceDelta={(trackingOriginGuard != null ? trackingOriginGuard.CurrentTrackingSpaceWorldDelta.ToString("F3") : "n/a")} "
            + $"sceneRootResolved={(_sceneRoot != null)} stableChecks={_stableHudChecks}/{Mathf.Max(1, stableChecksBeforeAnchorMount)} "
            + $"anchorPath='{(_panelAnchorTf != null ? GetTransformPath(_panelAnchorTf) : "<none>")}' "
            + $"rendererId={panelRenderer.GetInstanceID()}"
        );
    }

    private static bool IsInViewerViewport(Camera viewerCam, Vector3 worldPos)
    {
        if (viewerCam == null)
            return false;
        var vp = viewerCam.WorldToViewportPoint(worldPos);
        return vp.z > 0f && vp.x >= 0f && vp.x <= 1f && vp.y >= 0f && vp.y <= 1f;
    }

    private bool TryGetLink67AnchorPose(out Vector3 anchorPosWorld, out Quaternion anchorRotWorld)
    {
        anchorPosWorld = Vector3.zero;
        anchorRotWorld = Quaternion.identity;
        if (!TryResolveLinkAnchors())
            return false;

        float t = Mathf.Clamp01(link67MidBlend);
        anchorPosWorld = Vector3.Lerp(_link6Tf.position, _link7Tf.position, t);
        anchorRotWorld = link67UseLink7Rotation ? _link7Tf.rotation : Quaternion.Slerp(_link6Tf.rotation, _link7Tf.rotation, t);
        return true;
    }

    private bool TryApplyViewerFallbackPose(
        Transform panelTf,
        Renderer panelRenderer,
        Camera viewerCam,
        Vector3 viewerOffset,
        string poseSource,
        string warningMessage
    )
    {
        if (panelTf == null || viewerCam == null)
            return false;

        if (panelTf.parent != null)
            panelTf.SetParent(null, worldPositionStays: true);

        if (!panelTf.gameObject.activeSelf)
            panelTf.gameObject.SetActive(true);
        if (panelRenderer != null && !panelRenderer.enabled)
            panelRenderer.enabled = true;

        _lastPoseSource = poseSource;
        if (!_loggedPoseFallback)
        {
            Debug.LogWarning(warningMessage);
            _loggedPoseFallback = true;
        }

        Vector3 fallbackPos =
            viewerCam.transform.position
            + viewerCam.transform.right * viewerOffset.x
            + viewerCam.transform.up * viewerOffset.y
            + viewerCam.transform.forward * viewerOffset.z;
        Vector3 toViewer = viewerCam.transform.position - fallbackPos;
        Quaternion fallbackRot = toViewer.sqrMagnitude > 1e-8f
            ? Quaternion.LookRotation(toViewer.normalized, Vector3.up) * Quaternion.Euler(viewerFacingEulerOffset)
            : Quaternion.identity;

        panelTf.position = fallbackPos;
        panelTf.rotation = fallbackRot;
        if (setPanelScaleOnAttach)
            panelTf.localScale = panelScale;

        return true;
    }

    private void ApplyPoseFollow(Transform panelTf, bool hasPose, Vector3 camPos, Quaternion camRot)
    {
        if (panelTf == null)
            return;

        Camera viewerCam = ResolveViewerCamera();
        var panelRenderer = panelTf.GetComponent<Renderer>();

        if (forceViewerHudMode && viewerCam != null)
        {
            ApplyViewerHudPose(panelTf, panelRenderer, viewerCam, "viewer-hud");
            return;
        }

        if (ShouldHoldPanelInViewerHud(viewerCam, out var stabilizationReason))
        {
            if (viewerCam != null && ApplyViewerHudPose(
                panelTf,
                panelRenderer,
                viewerCam,
                "viewer-hud-stabilizing"
            ))
            {
                return;
            }

            _lastPoseSource = "viewer-hud-stabilizing";
            if (!_loggedStartupHudHold)
            {
                Debug.LogWarning(
                    "[SimPubRgbdSubscriber] Wrist panel stabilization requested viewer HUD mode, "
                    + $"but no viewer camera is available yet (reason='{stabilizationReason}')."
                );
                _loggedStartupHudHold = true;
            }
            return;
        }

        if (parentPanelUnderAnchorLink)
        {
            if (TryResolvePanelAnchorTransform(out var anchorTf))
            {
                if (panelTf.parent != anchorTf)
                    panelTf.SetParent(anchorTf, worldPositionStays: false);

                panelTf.localPosition = panelLocalOffsetOnAnchor;
                panelTf.localRotation = Quaternion.Euler(panelLocalEulerOnAnchor);
                if (setPanelScaleOnAttach)
                    panelTf.localScale = panelScale;

                if (!panelTf.gameObject.activeSelf)
                    panelTf.gameObject.SetActive(true);
                if (panelRenderer != null)
                    panelRenderer.enabled = true;

                _loggedAnchorWaiting = false;
                _loggedPoseFallback = false;
                _lastPoseSource = _panelAnchorUsedFallback ? "anchor-link7" : "anchor-hand";

                if (!_loggedFirstAnchorAppliedPose)
                {
                    Debug.Log(
                        $"[SimPubRgbdSubscriber] Panel mounted under anchor='{_panelAnchorResolvedName}' "
                        + $"path='{GetTransformPath(anchorTf)}' localPos={panelTf.localPosition} "
                        + $"localRot={panelTf.localRotation.eulerAngles} localScale={panelTf.localScale}"
                    );
                    _loggedFirstAnchorAppliedPose = true;
                }
            }
            else
            {
                if (enableViewerFallbackWhenPoseMissing
                    && viewerCam != null
                    && TryApplyViewerFallbackPose(
                    panelTf,
                    panelRenderer,
                    viewerCam,
                    viewerFallbackOffset,
                    "viewer-fallback",
                    $"[SimPubRgbdSubscriber] Waiting for panel anchor '{panelAnchorLinkName}' "
                    + $"(fallback '{panelAnchorFallbackLinkName}'); using viewer fallback panel placement to keep the wrist panel visible."
                ))
                {
                    _loggedAnchorWaiting = false;
                    return;
                }

                _lastPoseSource = enableViewerFallbackWhenPoseMissing ? "viewer-fallback" : "anchor-pending";
                if (!_loggedAnchorWaiting)
                {
                    Debug.LogWarning(
                        $"[SimPubRgbdSubscriber] Waiting for panel anchor '{panelAnchorLinkName}' "
                        + $"(fallback '{panelAnchorFallbackLinkName}'); "
                        + (enableViewerFallbackWhenPoseMissing
                            ? "viewer camera is unavailable, so the panel is keeping its current pose."
                            : "viewer fallback is disabled, so the panel is keeping its current pose until the anchor resolves.")
                    );
                    _loggedAnchorWaiting = true;
                }

                if (!panelTf.gameObject.activeSelf)
                    panelTf.gameObject.SetActive(true);
                if (panelRenderer != null)
                    panelRenderer.enabled = true;
            }
            return;
        }

        Vector3 poseBasePos = camPos;
        Quaternion poseBaseRot = camRot;
        bool poseBaseIsSceneLocal = true;
        bool usingLinkAnchorPose = false;
        if (anchorToLink67)
        {
            if (TryGetLink67AnchorPose(out var linkAnchorPosWorld, out var linkAnchorRotWorld))
            {
                if (parentRendererToSimScene && _sceneRoot != null && followPoseIsSceneLocal)
                {
                    poseBasePos = _sceneRoot.InverseTransformPoint(linkAnchorPosWorld);
                    poseBaseRot = Quaternion.Inverse(_sceneRoot.rotation) * linkAnchorRotWorld;
                    poseBaseIsSceneLocal = true;
                }
                else
                {
                    poseBasePos = linkAnchorPosWorld;
                    poseBaseRot = linkAnchorRotWorld;
                    poseBaseIsSceneLocal = false;
                }
                hasPose = true;
                usingLinkAnchorPose = true;
            }
            else
            {
                // In link67 mode, do not use raw wrist camera pose when anchor is unresolved.
                // That pose is often in a different frame and causes large placement offsets.
                hasPose = false;
                if (!_loggedAnchorWaiting)
                {
                    Debug.Log(
                        "[SimPubRgbdSubscriber] Waiting for link6/link7 anchor; using viewer fallback until the panel anchor resolves."
                    );
                    _loggedAnchorWaiting = true;
                }
            }
        }
        else if (_loggedAnchorWaiting)
        {
            _loggedAnchorWaiting = false;
        }

        if (!panelTf.gameObject.activeSelf)
            panelTf.gameObject.SetActive(true);
        if (panelRenderer != null && !panelRenderer.enabled)
            panelRenderer.enabled = true;

        if (!hasPose)
        {
            if (viewerCam != null && TryApplyViewerFallbackPose(
                panelTf,
                panelRenderer,
                viewerCam,
                viewerFallbackOffset,
                "viewer-fallback",
                "[SimPubRgbdSubscriber] Wrist pose/link67 anchor missing; using viewer fallback panel placement to keep the panel visible."
            ))
            {
                return;
            }

            _lastPoseSource = "viewer-fallback";
            if (!_loggedPoseFallback)
            {
                Debug.LogWarning(
                    "[SimPubRgbdSubscriber] Wrist pose/link67 anchor missing and viewer camera is unavailable; keeping the current panel pose."
                );
                _loggedPoseFallback = true;
            }
            if (!panelTf.gameObject.activeSelf)
                panelTf.gameObject.SetActive(true);
            if (panelRenderer != null)
                panelRenderer.enabled = true;
            return;
        }
        else if (_loggedPoseFallback)
        {
            _loggedPoseFallback = false;
        }

        _lastPoseSource = usingLinkAnchorPose ? "anchor-link7" : "anchor-hand";

        // no-op: poseBase is already chosen above (wrist pose or link67 anchor pose).

        if (parentRendererToSimScene)
        {
            if (_sceneRoot == null)
                TryResolveSceneRoot();
            if (_sceneRoot != null && panelTf.parent != _sceneRoot)
            {
                panelTf.SetParent(_sceneRoot, worldPositionStays: false);
                if (setPanelScaleOnAttach)
                    panelTf.localScale = panelScale;
            }
        }

        Vector3 offsetForPoseBase = panelOffsetLocal;
        if (!panelOffsetInWorldAxes)
        {
            // Legacy behavior: offset follows anchor local axes.
            offsetForPoseBase = poseBaseRot * panelOffsetLocal;
        }
        else if (poseBaseIsSceneLocal && _sceneRoot != null)
        {
            // Convert world-axis offset into scene-local space before adding to local pose base.
            offsetForPoseBase = _sceneRoot.InverseTransformVector(panelOffsetLocal);
        }

        Vector3 desiredPos = poseBasePos + offsetForPoseBase;
        Vector3 frontCamTargetPos = _frontCameraPos;
        if (!poseBaseIsSceneLocal && _sceneRoot != null)
            frontCamTargetPos = _sceneRoot.TransformPoint(_frontCameraPos);

        if (anchorToLink67)
        {
            if (Mathf.Abs(link67WorldLift) > 1e-5f)
            {
                if (parentRendererToSimScene && _sceneRoot != null)
                    desiredPos += _sceneRoot.InverseTransformVector(Vector3.up * link67WorldLift);
                else
                    desiredPos += Vector3.up * link67WorldLift;
            }

            if (_hasFrontCameraPose && Mathf.Abs(link67OutwardFromFront) > 1e-5f)
            {
                Vector3 toFrontForOffset = frontCamTargetPos - desiredPos;
                if (toFrontForOffset.sqrMagnitude > 1e-8f)
                    desiredPos += (-toFrontForOffset.normalized) * link67OutwardFromFront;
            }
        }

        Quaternion desiredRot;
        if (orientTowardViewer && viewerCam != null)
        {
            Vector3 viewerPos = viewerCam.transform.position;
            if (poseBaseIsSceneLocal && _sceneRoot != null)
                viewerPos = _sceneRoot.InverseTransformPoint(viewerPos);
            Vector3 toViewer = viewerPos - desiredPos;
            if (flattenViewerFacing)
            {
                var flattened = Vector3.ProjectOnPlane(toViewer, Vector3.up);
                if (flattened.sqrMagnitude > 1e-8f)
                    toViewer = flattened;
            }
            if (toViewer.sqrMagnitude > 1e-8f)
                desiredRot = Quaternion.LookRotation(toViewer.normalized, Vector3.up) * Quaternion.Euler(viewerFacingEulerOffset);
            else
                desiredRot = poseBaseRot * Quaternion.Euler(panelEulerOffset);
        }
        else if (orientTowardFrontCamera && _hasFrontCameraPose)
        {
            Vector3 toFrontCam = frontCamTargetPos - desiredPos;
            if (toFrontCam.sqrMagnitude > 1e-8f)
                desiredRot = Quaternion.LookRotation(toFrontCam.normalized, Vector3.up) * Quaternion.Euler(frontCameraFacingEulerOffset);
            else
                desiredRot = poseBaseRot * Quaternion.Euler(panelEulerOffset);
        }
        else
        {
            desiredRot = poseBaseRot * Quaternion.Euler(panelEulerOffset);
        }

        if (parentRendererToSimScene && _sceneRoot != null)
        {
            if (poseBaseIsSceneLocal)
            {
                if (calibrateOffsetFromTargetAtRuntime && !_calibratedOffsetFromTarget)
                {
                    Vector3 targetPos = calibrationTargetPosition;
                    if (!calibrationTargetIsSceneLocal)
                        targetPos = _sceneRoot.InverseTransformPoint(calibrationTargetPosition);

                    Vector3 delta = targetPos - desiredPos;
                    if (delta.sqrMagnitude > 1e-10f)
                    {
                        if (!panelOffsetInWorldAxes)
                            panelOffsetLocal += Quaternion.Inverse(poseBaseRot) * delta;
                        else
                            panelOffsetLocal += _sceneRoot.TransformVector(delta);

                        desiredPos = targetPos;
                    }

                    _calibratedOffsetFromTarget = true;
                    Debug.Log($"[SimPubRgbdSubscriber] Calibrated panel offset from target. newOffset={panelOffsetLocal} targetPos={targetPos}");
                }

                panelTf.localPosition = desiredPos;
                panelTf.localRotation = desiredRot;
            }
            else if (followPoseIsSceneLocal)
            {
                panelTf.localPosition = _sceneRoot.InverseTransformPoint(desiredPos);
                panelTf.localRotation = Quaternion.Inverse(_sceneRoot.rotation) * desiredRot;
            }
            else
            {
                panelTf.position = desiredPos;
                panelTf.rotation = desiredRot;
            }
        }
        else
        {
            // If scene root isn't resolved yet, treat desired pose as world-space to avoid
            // accidental placement relative to unrelated transforms.
            panelTf.position = desiredPos;
            panelTf.rotation = desiredRot;
        }

        if (setPanelScaleOnAttach)
            panelTf.localScale = panelScale;

        if (rescuePanelWhenOutOfView && viewerCam != null)
        {
            bool inView = IsInViewerViewport(viewerCam, panelTf.position);
            if (!inView)
            {
                Vector3 rescuePos =
                    viewerCam.transform.position
                    + viewerCam.transform.right * rescueViewerOffset.x
                    + viewerCam.transform.up * rescueViewerOffset.y
                    + viewerCam.transform.forward * rescueViewerOffset.z;
                Vector3 toViewer = viewerCam.transform.position - rescuePos;
                Quaternion rescueRot = toViewer.sqrMagnitude > 1e-8f
                    ? Quaternion.LookRotation(toViewer.normalized, Vector3.up) * Quaternion.Euler(viewerFacingEulerOffset)
                    : Quaternion.identity;

                panelTf.position = rescuePos;
                panelTf.rotation = rescueRot;
                if (!_panelRescueActive)
                {
                    _panelRescueActive = true;
                    Debug.Log($"[SimPubRgbdSubscriber] Panel rescued to viewer at pos={panelTf.position} rot={panelTf.rotation.eulerAngles}");
                }
            }
            else if (_panelRescueActive)
                {
                    _panelRescueActive = false;
                    Debug.Log("[SimPubRgbdSubscriber] Panel returned to anchor/frustum visibility.");
                }
        }

        if (usingLinkAnchorPose && !_loggedFirstAnchorAppliedPose)
        {
            _loggedFirstAnchorAppliedPose = true;
            Debug.Log(
                $"[SimPubRgbdSubscriber] Anchor-applied panel pose pos={panelTf.position} rot={panelTf.rotation.eulerAngles} scale={panelTf.localScale}"
            );
        }

        if (logPoseFollow)
        {
            Debug.Log(
                $"[SimPubRgbdSubscriber] PoseFollow cam='{followCameraName}' "
                + $"panelPos={panelTf.position} panelRot={panelTf.rotation.eulerAngles}"
            );
        }
        else if (!_loggedFirstPanelPose)
        {
            _loggedFirstPanelPose = true;
            Debug.Log(
                $"[SimPubRgbdSubscriber] Panel placed once at pos={panelTf.position} rot={panelTf.rotation.eulerAngles} scale={panelTf.localScale}"
            );
        }
    }
}
