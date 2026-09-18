using NetMQ;
using NetMQ.Sockets;
using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Threading;
using UnityEngine;
using IRIS.SceneLoader;
using Stopwatch = System.Diagnostics.Stopwatch;

public class SimPubPointCloudSubscriber : MonoBehaviour
{
    public string publisherIp = "127.0.0.1";
    public int topicPort = 7741;

    public string[] topics = new[] { "SimPub/Sensors/top/pc" };

    [Header("Render")]
    public SimplePointCloudRenderer pointCloudRenderer;
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
    public bool syncToSimSceneTransform = true;
    public bool logAttachInfo = true;

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

    private readonly ConcurrentQueue<Action> _mainThread = new();
    private Vector3[] _points;
    private Vector3[] _rawPoints;
    private Color32[] _colors;
    private int _lastPointCount;
    private int _frameCounter;
    private bool _attachedToSceneRoot;
    private Transform _sceneRoot;
    private readonly NetMQMessage _netMsg = new NetMQMessage();
    private bool _idleCloudCleared;
    private long _lastPayloadTick;
    private float _nextSceneCleanupTime;
    private float _nextReconnectTime;
    private static readonly double TimestampToSeconds = 1.0 / Stopwatch.Frequency;

    private const int BYTES_PER_POINT = 28; // 7 floats

    void Start()
    {
        AsyncIO.ForceDotNet.Force();

        if (pointCloudRenderer == null)
            pointCloudRenderer = GetComponent<SimplePointCloudRenderer>();

        if (pointCloudRenderer == null)
            Debug.LogWarning("[SimPubPointCloudSubscriber] No SimplePointCloudRenderer found. Add one to the same GameObject.");

        _lastPayloadTick = Stopwatch.GetTimestamp();
        TryAttachToSceneRoot();
        StartNetMq();
    }

    private void OnMsg(object sender, NetMQSocketEventArgs e)
    {
        if (_shuttingDown) return;

        _netMsg.Clear();
        var msg = _netMsg;
        // Guard against the shared NetMQ context being terminated during scene unload
        // (unhandled here -> poller-thread abort under IL2CPP).
        bool got;
        try { got = e.Socket.TryReceiveMultipartMessage(ref msg); }
        catch (TerminatingException) { _shuttingDown = true; try { e.Socket.ReceiveReady -= OnMsg; } catch { } return; }
        catch (ObjectDisposedException) { _shuttingDown = true; try { e.Socket.ReceiveReady -= OnMsg; } catch { } return; }
        if (!got) return;
        if (msg.FrameCount < 2) return;

        string topic;
        try { topic = msg[0].ConvertToString(); }
        catch { return; }

        // Payload is usually the last frame (safe even if frameCount > 2)
        var payload = msg[msg.FrameCount - 1].ToByteArray();
        if (payload == null || payload.Length < BYTES_PER_POINT) return;

        if (!topic.StartsWith("SimPub/Sensors/") || !topic.EndsWith("/pc")) return;

        int pointCount = Mathf.Min(payload.Length / BYTES_PER_POINT, maxPoints);
        if (pointCount <= 0) return;

        Interlocked.Exchange(ref _lastPayloadTick, Stopwatch.GetTimestamp());
        _idleCloudCleared = false;
        _mainThread.Enqueue(() => ApplyPoints(payload, pointCount));
    }

    private void ApplyPoints(byte[] payload, int pointCount)
    {
        if (pointCloudRenderer == null) return;

        if (pointCount != _lastPointCount)
        {
            _points = new Vector3[pointCount];
            _rawPoints = new Vector3[pointCount];
            _colors = new Color32[pointCount];
            _lastPointCount = pointCount;
        }

        for (int i = 0; i < pointCount; i++)
        {
            int o = i * BYTES_PER_POINT;

            float x = BitConverter.ToSingle(payload, o + 0) * positionScale;
            float y = BitConverter.ToSingle(payload, o + 4) * positionScale;
            float z = BitConverter.ToSingle(payload, o + 8) * positionScale;

            float r;
            float g;
            float b;
            if (forceDebugColor)
            {
                r = debugColor.r;
                g = debugColor.g;
                b = debugColor.b;
            }
            else
            {
                r = Mathf.Clamp01(BitConverter.ToSingle(payload, o + 12));
                g = Mathf.Clamp01(BitConverter.ToSingle(payload, o + 16));
                b = Mathf.Clamp01(BitConverter.ToSingle(payload, o + 20));
            }
            // Size is in payload but MeshTopology.Points uses shader size; we ignore s here.

            _rawPoints[i] = new Vector3(x, y, z);
            _colors[i] = new Color(r, g, b, 1f);
        }

        if (debugRecenterToAnchor)
        {
            Vector3 anchorLocal = ResolveDebugAnchorLocal();
            Vector3 sum = Vector3.zero;
            for (int i = 0; i < pointCount; i++)
                sum += _rawPoints[i];
            Vector3 centroid = sum / Mathf.Max(1, pointCount);
            for (int i = 0; i < pointCount; i++)
                _points[i] = (_rawPoints[i] - centroid) + anchorLocal + debugOffset;
        }
        else
        {
            for (int i = 0; i < pointCount; i++)
                _points[i] = _rawPoints[i] + debugOffset;
        }

        pointCloudRenderer.UpdateCloud(_points, _colors);

        if (logPointCount)
        {
            _frameCounter++;
            int n = Mathf.Max(1, logEveryNFrames);
            if ((_frameCounter % n) == 0)
            {
                Debug.Log($"[SimPubPointCloudSubscriber] points={pointCount} topic={string.Join(",", topics)} offset={debugOffset} recenter={debugRecenterToAnchor} anchor={ResolveDebugAnchorLocal()}");
            }
        }
    }

    void Update()
    {
        while (_mainThread.TryDequeue(out var a))
        {
            try { a(); } catch { }
        }

        if (destroyDuplicateSceneRoots)
            CleanupDuplicateSceneRoots();

        if (autoReconnectOnIdle && Time.unscaledTime >= _nextReconnectTime)
        {
            double idleForReconnect =
                (Stopwatch.GetTimestamp() - Interlocked.Read(ref _lastPayloadTick))
                * TimestampToSeconds;
            if (idleForReconnect >= Mathf.Max(0.5f, reconnectIdleSeconds))
            {
                Debug.LogWarning($"[SimPubPointCloudSubscriber] Idle for {idleForReconnect:F2}s. Reconnecting socket...");
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
                    Debug.Log($"[SimPubPointCloudSubscriber] Cleared cloud after {idleFor:F2}s idle.");
            }
        }

        if (_sceneRoot == null)
        {
            _attachedToSceneRoot = false;
        }
        else if (syncToSimSceneTransform)
        {
            transform.position = _sceneRoot.position;
            transform.rotation = _sceneRoot.rotation;
            transform.localScale = _sceneRoot.lossyScale;
        }

        if (!_attachedToSceneRoot)
            TryAttachToSceneRoot();
    }

    private void TryAttachToSceneRoot()
    {
        if (!parentToSimScene || _attachedToSceneRoot) return;

        // Only attach once the actual scene root exists.
        Transform root = null;
        if (SimSceneSpawner.Instance != null)
            root = SimSceneSpawner.Instance.GetSceneTransform(simSceneName);
        if (root == null)
        {
            var go = GameObject.Find(simSceneName);
            if (go != null) root = go.transform;
        }
        if (root == null) return;

        bool sceneRootChanged = _sceneRoot != null && _sceneRoot != root;
        _sceneRoot = root;
        if (sceneRootChanged && clearPointCloudOnSceneSwitch)
        {
            ClearPointCloud();
            if (logAttachInfo)
                Debug.Log("[SimPubPointCloudSubscriber] Cleared cloud on scene root switch.");
        }

        if (syncToSimSceneTransform)
        {
            transform.position = root.position;
            transform.rotation = root.rotation;
            transform.localScale = root.lossyScale;
        }
        _attachedToSceneRoot = true;
        if (logAttachInfo)
        {
            Debug.Log($"[SimPubPointCloudSubscriber] Bound to SimScene '{root.name}' (no re-parent).");
            Debug.Log($"[SimPubPointCloudSubscriber] Scene root pos={root.position} rot={root.rotation.eulerAngles} scale={root.lossyScale}");
            Debug.Log($"[SimPubPointCloudSubscriber] PC local pos={transform.localPosition} rot={transform.localRotation.eulerAngles} scale={transform.localScale}");
        }
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
        try
        {
            if (poller != null && poller.IsRunning)
                poller.Stop();   // blocking stop before socket dispose (avoids poller-thread "Must not be disposed" crash)
        }
        catch { }

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
        Debug.Log($"[SimPubPointCloudSubscriber] Connected tcp://{publisherIp}:{topicPort} topics=[{string.Join(", ", topics)}]");
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
        return transform.InverseTransformPoint(worldAnchor);
    }

    private void ClearPointCloud()
    {
        _points = null;
        _rawPoints = null;
        _colors = null;
        _lastPointCount = 0;
        pointCloudRenderer?.UpdateCloud(null, null);
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

        _sceneRoot = keep;
        _attachedToSceneRoot = true;
        if (removed > 0 && logAttachInfo)
            Debug.Log($"[SimPubPointCloudSubscriber] Removed {removed} duplicate scene root(s) for '{simSceneName}'.");
    }
}
