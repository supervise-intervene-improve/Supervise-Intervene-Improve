using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using IRIS.SceneLoader;
using NetMQ;
using NetMQ.Sockets;
using UnityEngine;
using Stopwatch = System.Diagnostics.Stopwatch;

public class GpuPointCloudSubscriber : MonoBehaviour
{
    public string publisherIp = "127.0.0.1";
    public int topicPort = 7741;
    public string[] topics = new[] { "SimPub/Sensors/top/pc" };

    [Header("Render")]
    public GpuInstancedPointCloudRenderer pointCloudRenderer;
    public int maxPoints = 40000;
    public float positionScale = 1.0f;

    [Header("Debug Visibility")]
    public bool forceDebugColor = false;
    public Color debugColor = new Color(0f, 1f, 0f, 1f);
    public Vector3 debugOffset = Vector3.zero;
    public bool debugRecenterToAnchor = false;
    public Vector3 debugAnchor = new Vector3(0f, 1.4f, 1.0f);
    public bool debugAnchorInFrontOfCamera = false;
    public float debugCameraDistance = 1.0f;
    public float debugCameraVerticalOffset = -0.1f;
    public bool logPointCount = true;
    public int logEveryNFrames = 60;

    [Header("Alignment")]
    public bool parentToSimScene = true;
    public string simSceneName = "MujocoScene";
    public bool syncToSimSceneTransform = false;
    public bool logAttachInfo = true;

    [Header("Visibility Fallback")]
    public bool preferSimplePointRendererOnAndroid = true;
    public Material simpleFallbackMaterial;
    public float simpleFallbackPointSize = 6.0f;
    public bool logSimpleFallbackActivation = true;

    [Header("Lifecycle")]
    public bool clearPointCloudOnSceneSwitch = true;
    public float clearPointCloudIfIdleSeconds = 1.0f;
    public bool destroyDuplicateSceneRoots = true;
    public float duplicateSceneCleanupInterval = 1.0f;
    public bool autoReconnectOnIdle = true;
    public float reconnectIdleSeconds = 8.0f;
    public float reconnectCooldownSeconds = 2.0f;

    private SubscriberSocket _sub;
    private NetMQPoller _poller;
    private volatile bool _shuttingDown;
    private readonly object _netMqLock = new object();
    private readonly object _frameLock = new object();
    private readonly NetMQMessage _netMsg = new NetMQMessage();

    private DecodedFrame _decodeFrame = new DecodedFrame();
    private DecodedFrame _pendingFrame;
    private DecodedFrame _renderFrame;

    private int _frameCounter;
    private bool _attachedToSceneRoot;
    private Transform _sceneRoot;
    private Transform _sceneAnchor;
    private string _sceneAnchorName;
    private bool _idleCloudCleared;
    private long _lastPayloadTick;
    private float _nextSceneCleanupTime;
    private float _nextReconnectTime;
    private int _contractWarningCount;
    private bool _loggedAttachDiagnostics;
    private bool _loggedWaitingForSceneAnchor;
    private bool _loggedLegacySyncWarning;
    private bool _loggedFirstDecodeDiagnostics;
    private bool _loggedFirstFrameDiagnostics;
    private bool _loggedRendererBlockedDiagnostics;
    private bool _loggedRendererBackendSelection;
    private bool _useSimpleFallbackRenderer;
    private SimplePointCloudRenderer _simpleFallbackRenderer;
    private Transform _simpleFallbackTransform;
    private Material _runtimeSimpleFallbackMaterial;
    private static readonly double TimestampToSeconds = 1.0 / Stopwatch.Frequency;

    private sealed class DecodedFrame
    {
        public string Topic;
        public Vector3[] RawPoints;
        public Vector3[] Points;
        public Color32[] Colors;
        public int Capacity;
        public int DeclaredCapacity;
        public int ActualCount;

        public void EnsureCapacity(int capacity)
        {
            capacity = Mathf.Max(1, capacity);
            if (RawPoints != null && Capacity >= capacity)
                return;

            RawPoints = new Vector3[capacity];
            Points = new Vector3[capacity];
            Colors = new Color32[capacity];
            Capacity = capacity;
        }
    }

    private void Start()
    {
        AsyncIO.ForceDotNet.Force();

        if (pointCloudRenderer == null)
            pointCloudRenderer = GetComponent<GpuInstancedPointCloudRenderer>();
        if (pointCloudRenderer == null)
            Debug.LogWarning("[GpuPointCloudSubscriber] No GpuInstancedPointCloudRenderer found. Add one to the same GameObject.");

        if (syncToSimSceneTransform && !_loggedLegacySyncWarning)
        {
            _loggedLegacySyncWarning = true;
            Debug.LogWarning(
                "[GpuPointCloudSubscriber] Legacy world-sync mode is deprecated and ignored. "
                + "Point clouds now use an explicit scene-anchor pose matrix."
            );
        }

        _lastPayloadTick = Stopwatch.GetTimestamp();
        ConfigureRendererBackend();
        TryAttachToSceneRoot();
        StartNetMq();
    }

    private void OnMsg(object sender, NetMQSocketEventArgs e)
    {
        if (_shuttingDown)
            return;

        _netMsg.Clear();
        var msg = _netMsg;
        // Guard against the shared NetMQ context being terminated during scene unload
        // (unhandled here -> poller-thread abort under IL2CPP).
        bool got;
        try { got = e.Socket.TryReceiveMultipartMessage(ref msg); }
        catch (TerminatingException) { _shuttingDown = true; try { e.Socket.ReceiveReady -= OnMsg; } catch { } return; }
        catch (ObjectDisposedException) { _shuttingDown = true; try { e.Socket.ReceiveReady -= OnMsg; } catch { } return; }
        if (!got || msg.FrameCount < 2)
            return;

        string topic;
        try { topic = msg[0].ConvertToString(); }
        catch { return; }

        if (!topic.StartsWith("SimPub/Sensors/") || !topic.EndsWith("/pc"))
            return;

        var payload = msg[msg.FrameCount - 1].ToByteArray();
        if (!GpuPointCloudContract.TryParseHeader(payload, out var header, out var error))
        {
            LogContractWarning(error);
            return;
        }

        if (header.DeclaredCapacity > maxPoints)
        {
            LogContractWarning(
                $"Dropped GPU point-cloud frame for '{topic}': declared capacity {header.DeclaredCapacity} exceeds subscriber maxPoints {maxPoints}."
            );
            return;
        }

        var frame = _decodeFrame ?? new DecodedFrame();
        frame.EnsureCapacity(Mathf.Max(1, header.DeclaredCapacity));
        if (!GpuPointCloudContract.TryDecodeInto(
                payload,
                header,
                frame.RawPoints,
                frame.Colors,
                positionScale,
                forceDebugColor,
                debugColor,
                out error))
        {
            LogContractWarning(error);
            return;
        }

        frame.Topic = topic;
        frame.DeclaredCapacity = header.DeclaredCapacity;
        frame.ActualCount = header.ActualCount;

        if (!_loggedFirstDecodeDiagnostics)
        {
            _loggedFirstDecodeDiagnostics = true;
            Vector3 representativeRaw = frame.ActualCount > 0 ? frame.RawPoints[0] : Vector3.zero;
            Debug.Log(
                "[GpuPointCloudSubscriber] First decoded frame "
                + $"topic={topic} actual={frame.ActualCount} declared={frame.DeclaredCapacity} "
                + $"representativeRaw={representativeRaw}"
            );
        }

        Interlocked.Exchange(ref _lastPayloadTick, Stopwatch.GetTimestamp());
        _idleCloudCleared = false;

        lock (_frameLock)
        {
            var previousPending = _pendingFrame;
            _pendingFrame = frame;
            _decodeFrame = previousPending != null && !ReferenceEquals(previousPending, frame)
                ? previousPending
                : new DecodedFrame();
        }
    }

    private void Update()
    {
        if (destroyDuplicateSceneRoots)
            CleanupDuplicateSceneRoots();

        if (parentToSimScene
            && (_sceneRoot == null || _sceneAnchor == null || _sceneAnchor.parent != _sceneRoot))
        {
            _attachedToSceneRoot = false;
        }

        if (!_attachedToSceneRoot)
            TryAttachToSceneRoot();

        UpdateRendererPose();

        ApplyPendingFrame();

        if (autoReconnectOnIdle && Time.unscaledTime >= _nextReconnectTime)
        {
            double idleForReconnect =
                (Stopwatch.GetTimestamp() - Interlocked.Read(ref _lastPayloadTick))
                * TimestampToSeconds;
            if (idleForReconnect >= Mathf.Max(0.5f, reconnectIdleSeconds))
            {
                Debug.LogWarning($"[GpuPointCloudSubscriber] Idle for {idleForReconnect:F2}s. Reconnecting socket...");
                ReconnectNetMq();
            }
        }

        if (!_idleCloudCleared && clearPointCloudIfIdleSeconds > 0f)
        {
            double idleFor =
                (Stopwatch.GetTimestamp() - Interlocked.Read(ref _lastPayloadTick))
                * TimestampToSeconds;
            if (idleFor >= clearPointCloudIfIdleSeconds)
            {
                ClearPointCloud();
                _idleCloudCleared = true;
                if (logPointCount)
                    Debug.Log($"[GpuPointCloudSubscriber] Cleared cloud after {idleFor:F2}s idle.");
            }
        }

    }

    private void ApplyPendingFrame()
    {
        DecodedFrame frame = null;
        lock (_frameLock)
        {
            if (_pendingFrame != null)
            {
                frame = _pendingFrame;
                _pendingFrame = null;
            }
        }

        if (frame == null)
            return;

        _renderFrame = frame;

        if (parentToSimScene && _sceneAnchor == null)
        {
            if (!_loggedWaitingForSceneAnchor)
            {
                _loggedWaitingForSceneAnchor = true;
                Debug.LogWarning(
                    $"[GpuPointCloudSubscriber] Waiting for scene anchor under '{simSceneName}' before rendering '{string.Join(", ", topics)}'."
                );
            }
            return;
        }

        if (frame.ActualCount <= 0)
        {
            ClearPointCloud();
            return;
        }

        if (debugRecenterToAnchor)
        {
            Vector3 anchorLocal = ResolveDebugAnchorLocal();
            Vector3 sum = Vector3.zero;
            for (int i = 0; i < frame.ActualCount; i++)
                sum += frame.RawPoints[i];

            Vector3 centroid = sum / Mathf.Max(1, frame.ActualCount);
            for (int i = 0; i < frame.ActualCount; i++)
                frame.Points[i] = (frame.RawPoints[i] - centroid) + anchorLocal + debugOffset;
        }
        else
        {
            for (int i = 0; i < frame.ActualCount; i++)
                frame.Points[i] = frame.RawPoints[i] + debugOffset;
        }

        UpdateRendererPose();
        if (_useSimpleFallbackRenderer)
        {
            _simpleFallbackRenderer?.UpdateCloud(frame.Points, frame.Colors);
        }
        else
        {
            pointCloudRenderer?.SetPointCloud(frame.Points, frame.Colors, frame.ActualCount, frame.DeclaredCapacity);
        }
        LogFirstFrameDiagnostics(frame);

        if (logPointCount)
        {
            _frameCounter++;
            int n = Mathf.Max(1, logEveryNFrames);
            if ((_frameCounter % n) == 0)
            {
                Debug.Log(
                    $"[GpuPointCloudSubscriber] points={frame.ActualCount}/{frame.DeclaredCapacity} "
                    + $"topic={string.Join(",", topics)} offset={debugOffset} recenter={debugRecenterToAnchor} "
                    + $"anchor={ResolveDebugAnchorLocal()}"
                );
            }
        }
    }

    private void LogFirstFrameDiagnostics(DecodedFrame frame)
    {
        if (!_loggedFirstFrameDiagnostics)
        {
            _loggedFirstFrameDiagnostics = true;
            Vector3 representativeRaw = frame.RawPoints[0];
            Vector3 representativeRendered = frame.Points[0];
            string rendererState = pointCloudRenderer == null
                ? "missing"
                : (pointCloudRenderer.RuntimeReady ? "ready" : "blocked");
            string runtimeBlockReason = pointCloudRenderer == null
                ? "no-renderer"
                : pointCloudRenderer.RuntimeBlockReason;
            Bounds rendererBounds;
            Matrix4x4 renderPoseMatrix;
            string activeRendererMode;
            if (_useSimpleFallbackRenderer && _simpleFallbackTransform != null)
            {
                activeRendererMode = "simple-fallback";
                rendererState = "simple-fallback";
                runtimeBlockReason = string.Empty;
                renderPoseMatrix = _simpleFallbackTransform.localToWorldMatrix;
                rendererBounds = new Bounds(_simpleFallbackTransform.position, Vector3.one * 1000f);
            }
            else
            {
                activeRendererMode = "gpu-instanced";
                rendererBounds = pointCloudRenderer != null
                    ? pointCloudRenderer.CurrentRenderBounds
                    : new Bounds(transform.position, Vector3.zero);
                renderPoseMatrix = pointCloudRenderer != null
                    ? pointCloudRenderer.CurrentRenderPoseMatrix
                    : transform.localToWorldMatrix;
            }
            string anchorPath = _sceneAnchor != null ? GetTransformPath(_sceneAnchor) : "<none>";
            Vector3 anchorLocalPos = _sceneAnchor != null ? _sceneAnchor.localPosition : Vector3.zero;
            Vector3 anchorLocalScale = _sceneAnchor != null ? _sceneAnchor.localScale : Vector3.one;
            Vector3 anchorLocalEuler = _sceneAnchor != null ? _sceneAnchor.localEulerAngles : Vector3.zero;
            Vector3 anchorWorldPos = _sceneAnchor != null ? _sceneAnchor.position : transform.position;
            Vector3 anchorWorldScale = _sceneAnchor != null ? _sceneAnchor.lossyScale : transform.lossyScale;
            Vector3 anchorWorldEuler = _sceneAnchor != null ? _sceneAnchor.rotation.eulerAngles : transform.rotation.eulerAngles;

            Debug.Log(
                "[GpuPointCloudSubscriber] First frame diagnostics "
                + $"topic={frame.Topic} actual={frame.ActualCount} declared={frame.DeclaredCapacity} "
                + $"rendererMode={activeRendererMode} "
                + $"representativeRaw={representativeRaw} representativeRendered={representativeRendered} "
                + $"anchorPath='{anchorPath}' anchorLocalPos={anchorLocalPos} anchorLocalRot={anchorLocalEuler} "
                + $"anchorLocalScale={anchorLocalScale} anchorWorldPos={anchorWorldPos} "
                + $"anchorWorldRot={anchorWorldEuler} anchorWorldScale={anchorWorldScale} "
                + $"rendererState={rendererState} rendererBoundsCenter={rendererBounds.center} "
                + $"rendererBoundsSize={rendererBounds.size} renderPoseMatrix={FormatMatrix(renderPoseMatrix)} "
                + $"blockReason='{runtimeBlockReason}'"
            );
        }

        if (pointCloudRenderer != null
            && !_useSimpleFallbackRenderer
            && !pointCloudRenderer.RuntimeReady
            && !_loggedRendererBlockedDiagnostics)
        {
            _loggedRendererBlockedDiagnostics = true;
            Debug.LogError(
                "[GpuPointCloudSubscriber] Successfully decoded a GPU point-cloud frame, but rendering is blocked: "
                + pointCloudRenderer.RuntimeBlockReason
            );
        }
    }

    private void LogContractWarning(string message)
    {
        if (_contractWarningCount < 10)
            Debug.LogWarning($"[GpuPointCloudSubscriber] {message}");
        _contractWarningCount++;
    }

    private void TryAttachToSceneRoot()
    {
        if (!parentToSimScene)
        {
            _attachedToSceneRoot = true;
            return;
        }

        if (_attachedToSceneRoot && _sceneRoot != null && _sceneAnchor != null && _sceneAnchor.parent == _sceneRoot)
            return;

        Transform root = ResolveSceneRoot();
        if (root == null)
            return;

        bool sceneRootChanged = _sceneRoot != null && _sceneRoot != root;
        _sceneRoot = root;
        if (sceneRootChanged && clearPointCloudOnSceneSwitch)
        {
            ClearPointCloud();
            if (logAttachInfo)
                Debug.Log("[GpuPointCloudSubscriber] Cleared cloud on scene root switch.");
        }

        if (sceneRootChanged)
            ResetAttachmentDiagnostics();

        _sceneAnchor = ResolveOrCreateSceneAnchor(root);
        if (_sceneAnchor == null)
            return;

        _attachedToSceneRoot = true;
        _loggedWaitingForSceneAnchor = false;
        UpdateRendererPose();

        if (logAttachInfo && !_loggedAttachDiagnostics)
        {
            _loggedAttachDiagnostics = true;
            Matrix4x4 renderPoseMatrix = pointCloudRenderer != null
                ? pointCloudRenderer.CurrentRenderPoseMatrix
                : _sceneAnchor.localToWorldMatrix;
            Debug.Log($"[GpuPointCloudSubscriber] Bound to SimScene '{root.name}' via anchor '{_sceneAnchor.name}'.");
            Debug.Log(
                $"[GpuPointCloudSubscriber] Anchor path='{GetTransformPath(_sceneAnchor)}' "
                + $"localPos={_sceneAnchor.localPosition} localRot={_sceneAnchor.localEulerAngles} localScale={_sceneAnchor.localScale}"
            );
            Debug.Log(
                $"[GpuPointCloudSubscriber] Anchor worldPos={_sceneAnchor.position} "
                + $"worldRot={_sceneAnchor.rotation.eulerAngles} worldScale={_sceneAnchor.lossyScale}"
            );
            Debug.Log($"[GpuPointCloudSubscriber] Render pose matrix={FormatMatrix(renderPoseMatrix)}");
        }
    }

    private Transform ResolveSceneRoot()
    {
        Transform root = null;
        if (SimSceneSpawner.Instance != null)
            root = SimSceneSpawner.Instance.GetSceneTransform(simSceneName);
        if (root == null)
        {
            var go = GameObject.Find(simSceneName);
            if (go != null)
                root = go.transform;
        }
        return root;
    }

    private Transform ResolveOrCreateSceneAnchor(Transform root)
    {
        if (root == null)
            return null;

        string anchorName = BuildSceneAnchorName();
        var anchor = root.Find(anchorName);
        if (anchor == null)
        {
            var anchorGo = new GameObject(anchorName);
            anchor = anchorGo.transform;
            anchor.SetParent(root, false);
        }

        anchor.localPosition = Vector3.zero;
        anchor.localRotation = Quaternion.identity;
        anchor.localScale = Vector3.one;
        return anchor;
    }

    private string BuildSceneAnchorName()
    {
        if (!string.IsNullOrEmpty(_sceneAnchorName))
            return _sceneAnchorName;

        string fragment = gameObject.name;
        if (topics != null)
        {
            for (int i = 0; i < topics.Length; i++)
            {
                string topic = topics[i];
                if (string.IsNullOrWhiteSpace(topic))
                    continue;

                string[] parts = topic.Split('/');
                if (parts.Length >= 3 && !string.IsNullOrWhiteSpace(parts[2]))
                {
                    fragment = parts[2];
                    break;
                }
            }
        }

        var sanitized = new StringBuilder(fragment.Length);
        for (int i = 0; i < fragment.Length; i++)
        {
            char c = fragment[i];
            sanitized.Append(char.IsLetterOrDigit(c) ? c : '_');
        }

        _sceneAnchorName = $"PointCloudAnchor_{sanitized}";
        return _sceneAnchorName;
    }

    private void UpdateRendererPose()
    {
        Matrix4x4 poseMatrix;
        Vector3 boundsCenter;
        if (pointCloudRenderer == null)
        {
            if (parentToSimScene)
            {
                if (_sceneAnchor == null)
                    return;
                poseMatrix = _sceneAnchor.localToWorldMatrix;
                boundsCenter = _sceneAnchor.position;
            }
            else
            {
                poseMatrix = transform.localToWorldMatrix;
                boundsCenter = transform.position;
            }
        }
        else
        {
            if (parentToSimScene)
            {
                if (_sceneAnchor == null)
                    return;
                poseMatrix = _sceneAnchor.localToWorldMatrix;
                boundsCenter = _sceneAnchor.position;
            }
            else
            {
                poseMatrix = transform.localToWorldMatrix;
                boundsCenter = transform.position;
            }

            pointCloudRenderer.SetRenderPose(poseMatrix, boundsCenter);
        }

        if (_useSimpleFallbackRenderer && _simpleFallbackTransform != null)
            ApplySimpleFallbackPose(poseMatrix);
    }

    private void OnApplicationQuit()
    {
        _shuttingDown = true;
        ClearPointCloud();
        ShutdownNetMQ();
    }

    private void OnDisable()
    {
        // Stop the NetMQ poller while the shared context is still alive (scene-exit proactive
        // shutdown), so it is gone before the XR framework node's OnDestroy terminates the context. Idempotent.
        _shuttingDown = true;
        ShutdownNetMQ();
    }

    private void OnDestroy()
    {
        _shuttingDown = true;
        ClearPointCloud();
        ShutdownNetMQ();
        if (_simpleFallbackTransform != null)
        {
            if (Application.isPlaying)
                Destroy(_simpleFallbackTransform.gameObject);
            else
                DestroyImmediate(_simpleFallbackTransform.gameObject);
        }
        if (_runtimeSimpleFallbackMaterial != null)
        {
            if (Application.isPlaying)
                Destroy(_runtimeSimpleFallbackMaterial);
            else
                DestroyImmediate(_runtimeSimpleFallbackMaterial);
            _runtimeSimpleFallbackMaterial = null;
        }
    }

    private void ShutdownNetMQ()
    {
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
        try { if (poller != null && poller.IsRunning) poller.Stop(); } catch { }   // blocking stop before socket dispose
        try { sub?.Close(); } catch { }
        try { sub?.Dispose(); } catch { }
        try { poller?.Dispose(); } catch { }
    }

    private void StartNetMq()
    {
        var sub = new SubscriberSocket();
        sub.Options.ReceiveHighWatermark = 2;
        sub.Options.Linger = TimeSpan.Zero;
        sub.Connect($"tcp://{publisherIp}:{topicPort}");
        foreach (var t in topics)
            sub.Subscribe(t);

        sub.ReceiveReady += OnMsg;
        var poller = new NetMQPoller { sub };
        lock (_netMqLock)
        {
            _sub = sub;
            _poller = poller;
        }
        poller.RunAsync();

        Interlocked.Exchange(ref _lastPayloadTick, Stopwatch.GetTimestamp());
        _nextReconnectTime = Time.unscaledTime + Mathf.Max(0.5f, reconnectCooldownSeconds);
        Debug.Log($"[GpuPointCloudSubscriber] Connected tcp://{publisherIp}:{topicPort} topics=[{string.Join(", ", topics)}]");
    }

    private void ReconnectNetMq()
    {
        ShutdownNetMQ();
        _shuttingDown = false;
        StartNetMq();
    }

    private Vector3 ResolveDebugAnchorLocal()
    {
        if (!debugAnchorInFrontOfCamera)
            return debugAnchor;

        var cam = Camera.main;
        if (cam == null)
            return debugAnchor;

        Vector3 worldAnchor =
            cam.transform.position
            + cam.transform.forward * debugCameraDistance
            + cam.transform.up * debugCameraVerticalOffset;
        if (_sceneAnchor != null)
            return _sceneAnchor.worldToLocalMatrix.MultiplyPoint3x4(worldAnchor);

        if (_useSimpleFallbackRenderer && _simpleFallbackTransform != null)
            return _simpleFallbackTransform.worldToLocalMatrix.MultiplyPoint3x4(worldAnchor);

        Matrix4x4 renderPose = pointCloudRenderer != null
            ? pointCloudRenderer.CurrentRenderPoseMatrix
            : transform.localToWorldMatrix;
        return renderPose.inverse.MultiplyPoint3x4(worldAnchor);
    }

    private void ClearPointCloud()
    {
        _renderFrame = null;
        pointCloudRenderer?.Clear();
        _simpleFallbackRenderer?.UpdateCloud(null, null);
    }

    private bool IsSimSceneRootName(string rootName)
    {
        if (string.IsNullOrWhiteSpace(simSceneName) || string.IsNullOrWhiteSpace(rootName))
            return false;
        if (string.Equals(rootName, simSceneName, StringComparison.Ordinal))
            return true;
        return rootName.StartsWith(simSceneName + "(", StringComparison.Ordinal);
    }

    private void CleanupDuplicateSceneRoots()
    {
        float interval = Mathf.Max(0.2f, duplicateSceneCleanupInterval);
        if (Time.unscaledTime < _nextSceneCleanupTime)
            return;
        _nextSceneCleanupTime = Time.unscaledTime + interval;

        var allTransforms = UnityEngine.Object.FindObjectsByType<Transform>(
            FindObjectsInactive.Include,
            FindObjectsSortMode.None
        );
        List<Transform> sceneRoots = null;
        for (int i = 0; i < allTransforms.Length; i++)
        {
            var tf = allTransforms[i];
            if (tf == null)
                continue;
            if (!IsSimSceneRootName(tf.name))
                continue;
            sceneRoots ??= new List<Transform>(2);
            sceneRoots.Add(tf);
        }

        if (sceneRoots == null || sceneRoots.Count <= 1)
            return;

        Transform keep = null;
        if (_sceneRoot != null && sceneRoots.Contains(_sceneRoot))
            keep = _sceneRoot;

        if (keep == null && SimSceneSpawner.Instance != null)
        {
            var spawnerRoot = SimSceneSpawner.Instance.GetSceneTransform(simSceneName);
            if (spawnerRoot != null && sceneRoots.Contains(spawnerRoot))
                keep = spawnerRoot;
        }

        if (keep == null)
            keep = sceneRoots[sceneRoots.Count - 1];

        int removed = 0;
        for (int i = 0; i < sceneRoots.Count; i++)
        {
            var root = sceneRoots[i];
            if (root == null || root == keep)
                continue;
            removed++;
            Destroy(root.gameObject);
        }

        bool keepChanged = _sceneRoot != keep;
        _sceneRoot = keep;
        if (keepChanged || _sceneAnchor == null || _sceneAnchor.parent != keep)
        {
            if (keepChanged && clearPointCloudOnSceneSwitch)
                ClearPointCloud();
            _sceneAnchor = null;
            _attachedToSceneRoot = false;
            ResetAttachmentDiagnostics();
        }
        else
        {
            _attachedToSceneRoot = true;
        }

        if (removed > 0 && logAttachInfo)
            Debug.Log($"[GpuPointCloudSubscriber] Removed {removed} duplicate scene root(s) for '{simSceneName}'.");
    }

    private void ResetAttachmentDiagnostics()
    {
        _loggedAttachDiagnostics = false;
        _loggedWaitingForSceneAnchor = false;
        _loggedFirstFrameDiagnostics = false;
        _loggedRendererBlockedDiagnostics = false;
    }

    private void ConfigureRendererBackend()
    {
        _useSimpleFallbackRenderer =
            preferSimplePointRendererOnAndroid
            && Application.platform == RuntimePlatform.Android;

        if (_useSimpleFallbackRenderer)
        {
            EnsureSimpleFallbackRenderer();
            if (_simpleFallbackRenderer == null)
            {
                _useSimpleFallbackRenderer = false;
                Debug.LogWarning(
                    $"[GpuPointCloudSubscriber] Simple fallback could not be initialized for '{name}'. "
                    + "Falling back to GPU instanced rendering."
                );
            }
        }

        if (!_loggedRendererBackendSelection)
        {
            _loggedRendererBackendSelection = true;
            string backend = _useSimpleFallbackRenderer ? "simple-fallback" : "gpu-instanced";
            Debug.Log(
                $"[GpuPointCloudSubscriber] Active renderer backend='{backend}' platform={Application.platform}."
            );
        }
    }

    private void EnsureSimpleFallbackRenderer()
    {
        if (_simpleFallbackRenderer != null)
            return;

        var fallbackGo = new GameObject($"{name}_SimpleFallback");
        fallbackGo.layer = gameObject.layer;
        _simpleFallbackTransform = fallbackGo.transform;
        _simpleFallbackTransform.SetParent(transform, false);
        _simpleFallbackTransform.localPosition = Vector3.zero;
        _simpleFallbackTransform.localRotation = Quaternion.identity;
        _simpleFallbackTransform.localScale = Vector3.one;

        fallbackGo.AddComponent<MeshFilter>();
        var meshRenderer = fallbackGo.AddComponent<MeshRenderer>();
        meshRenderer.sharedMaterial = ResolveSimpleFallbackMaterial();
        if (meshRenderer.sharedMaterial == null)
        {
            Debug.LogError(
                $"[GpuPointCloudSubscriber] Simple fallback for '{name}' has no material. "
                + "Assign simpleFallbackMaterial in the scene."
            );
            return;
        }

        _simpleFallbackRenderer = fallbackGo.AddComponent<SimplePointCloudRenderer>();
        _simpleFallbackRenderer.scale = 1.0f;
        _simpleFallbackRenderer.pointSize = Mathf.Max(1.0f, simpleFallbackPointSize);

        if (logSimpleFallbackActivation)
        {
            string materialName = meshRenderer.sharedMaterial != null
                ? meshRenderer.sharedMaterial.name
                : "<missing>";
            Debug.Log(
                "[GpuPointCloudSubscriber] Activated simple point-render fallback on Android "
                + $"for '{name}' using material '{materialName}' pointSize={_simpleFallbackRenderer.pointSize:F1}."
            );
        }
    }

    private Material ResolveSimpleFallbackMaterial()
    {
        if (simpleFallbackMaterial != null)
            return simpleFallbackMaterial;

        if (_runtimeSimpleFallbackMaterial != null)
            return _runtimeSimpleFallbackMaterial;

        Shader shader = Shader.Find("SII/PointCloud/UnlitPoints");
        if (shader == null)
        {
            Debug.LogError(
                "[GpuPointCloudSubscriber] Failed to find fallback shader 'SII/PointCloud/UnlitPoints'. "
                + "Assign simpleFallbackMaterial in the scene to force point-cloud visibility fallback."
            );
            return null;
        }

        _runtimeSimpleFallbackMaterial = new Material(shader)
        {
            name = $"{name}_RuntimePointCloudFallback"
        };
        _runtimeSimpleFallbackMaterial.SetColor("_Color", Color.white);
        _runtimeSimpleFallbackMaterial.SetFloat("_PointSize", Mathf.Max(1.0f, simpleFallbackPointSize));
        return _runtimeSimpleFallbackMaterial;
    }

    private void ApplySimpleFallbackPose(Matrix4x4 poseMatrix)
    {
        Vector3 position = poseMatrix.GetColumn(3);
        Vector3 forward = poseMatrix.GetColumn(2);
        Vector3 up = poseMatrix.GetColumn(1);
        if (forward.sqrMagnitude < 1e-6f || up.sqrMagnitude < 1e-6f)
            return;

        _simpleFallbackTransform.position = position;
        _simpleFallbackTransform.rotation = Quaternion.LookRotation(forward, up);
        _simpleFallbackTransform.localScale = Vector3.one;
    }

    private static string GetTransformPath(Transform tf)
    {
        if (tf == null)
            return "<none>";

        var names = new List<string>(8);
        for (Transform current = tf; current != null; current = current.parent)
            names.Add(current.name);
        names.Reverse();
        return string.Join("/", names);
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
