using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using IRIS.SceneLoader;
using NetMQ;
using NetMQ.Sockets;
using UnityEngine;
using UnityEngine.Rendering;

[DisallowMultipleComponent]
[RequireComponent(typeof(MeshFilter), typeof(MeshRenderer))]
public class GpuMergedPointCloudLoader : MonoBehaviour
{
    private const string LoaderBuildStamp = "2026-03-17-gpu-merged-draw";
    private const string SessionStateTopic = "SimPub/Status/session_state";
    private const float RuntimeBlockedLogIntervalSeconds = 5.0f;
    // How long a source may go without a payload before its points are removed.
    // Was a hard-coded 1.0f, which is far too aggressive for a --pc_round_robin publisher:
    // that renders ONE camera per tick, so a camera's refresh period is tick_dt x n_cameras.
    // Measured at 9 concurrent sessions that is ~350 ms normally but 1.2-3.0 s whenever the
    // publisher slows down, and every excursion past this value deletes that camera's cloud
    // until its next payload — the "cameras drop out one at a time then come back" loop.
    // 4 s keeps stale-data protection while tolerating publisher hiccups.
    [Tooltip("Seconds without a payload before a source's points are cleared. Must exceed " +
             "the publisher's worst per-camera refresh interval (see --pc_max_source_age_s).")]
    [SerializeField] private float sourceIdleClearSeconds = 4.0f;
    private static readonly int MatrixBufferId = Shader.PropertyToID("matrixBuffer");
    private static readonly int ColorBufferId = Shader.PropertyToID("colorBuffer");
    private static readonly int BaseColorId = Shader.PropertyToID("_BaseColor");
    private static readonly int OverlayAlphaId = Shader.PropertyToID("_OverlayAlpha");
    private static readonly int VisibilityGainId = Shader.PropertyToID("_VisibilityGain");
    private static readonly int MinVisibleBrightnessId = Shader.PropertyToID("_MinVisibleBrightness");
    private static readonly int SrcBlendId = Shader.PropertyToID("_SrcBlend");
    private static readonly int DstBlendId = Shader.PropertyToID("_DstBlend");
    private static readonly int ZWriteId = Shader.PropertyToID("_ZWrite");
    private static readonly int ZTestId = Shader.PropertyToID("_ZTest");
    private static readonly int RenderPoseMatrixId = Shader.PropertyToID("_RenderPoseMatrix");
    private static readonly double TimestampToSeconds = 1.0 / System.Diagnostics.Stopwatch.Frequency;

    [Header("Connection")]
    public string publisherIp = "127.0.0.1";
    public int topicPort = 7741;
    public int expectedSessionIndex = -1;
    public string[] topics = new[] { "SimPub/Sensors/top/pc", "SimPub/Sensors/right/pc", "SimPub/Sensors/left/pc" };
    public int maxPointsPerSource = 100000;
    public int maxCombinedPoints = 300000;
    public float positionScale = 1.0f;

    [Header("Rendering")]
    public Material instancedMaterial;
    public float CubeSize = 0.003f;
    public float scale = 1.0f;
    public bool renderEnabled = true;
    public float renderBoundsSize = 1000.0f;
    public Color fallbackBaseColor = new Color32(255, 255, 255, 242);
    public bool logBufferLifecycle = true;
    public PointCloudRenderMode renderMode = PointCloudRenderMode.DepthTestedOpaque;
    public bool useDepthTestDebugMode = false;
    public float debugCubeSizeOverride = 0.0f;

    [Header("Alignment")]
    public bool parentToSimScene = true;
    public string simSceneName = "MujocoScene";
    public string sceneAnchorNameOverride = string.Empty;
    public Vector3 debugOffset = Vector3.zero;
    public bool logAttachInfo = true;

    [Header("Lifecycle")]
    public bool autoReconnectOnIdle = true;
    public float reconnectIdleSeconds = 8.0f;
    public float reconnectCooldownSeconds = 2.0f;

    [Header("Startup Stabilization")]
    public bool waitForSceneAnchorBeforeFirstDraw = true;
    public bool requireStableTrackingBeforeFirstDraw = true;
    public int stableChecksBeforeFirstDraw = 3;

    [Header("Diagnostics")]
    public float heartbeatLogIntervalSeconds = 1.0f;

    private sealed class DecodedFrame
    {
        public Vector3[] RawPoints;
        public Color32[] Colors;
        public int Capacity;
        public int DeclaredCapacity;
        public int ActualCount;

        public void EnsureCapacity(int capacity)
        {
            capacity = Mathf.Max(1, capacity);
            if (RawPoints != null && Capacity >= capacity)
                return;

            // Grow with slack so a slowly-growing cloud does not reallocate every frame.
            capacity = Mathf.Max(capacity, Capacity * 2);
            RawPoints = new Vector3[capacity];
            Colors = new Color32[capacity];
            Capacity = capacity;
        }
    }

    private sealed class SourceState
    {
        public string Topic;
        public string Label;
        public DecodedFrame DecodeFrame = new DecodedFrame();
        public DecodedFrame PendingFrame;
        public DecodedFrame RenderFrame;
        public long LastPayloadTick;
        public int ContractWarningCount;
        public bool LoggedFirstFrameDiagnostics;
        public bool LoggedIdleClear;

        public long RxCount;
    }

    [Serializable]
    private sealed class SessionStateHeartbeat
    {
        public int version;
        public int session_index;
        public int topic_port;
        public string episode_id;
        public long policy_seq;
        public int frame_idx;
        public string mode;
        public bool paused;
        public bool intervention_live;
        public string intervention_phase;
        public bool upstream_stale;
        public float upstream_stale_age_s;
        public float qpos_delta_norm;
        public float state_change_age_s;
    }

    private readonly object _frameLock = new object();
    private readonly object _netMqLock = new object();
    private readonly NetMQMessage _netMessage = new NetMQMessage();
    private readonly Dictionary<string, SourceState> _sourceByTopic = new Dictionary<string, SourceState>(StringComparer.Ordinal);
    private readonly List<SourceState> _sourceList = new List<SourceState>();

    private SubscriberSocket _sub;
    private NetMQPoller _poller;
    private volatile bool _shuttingDown;
    private long _lastAnyPayloadTick;
    private float _nextReconnectTime;
    private string _topicSignature = string.Empty;
    private int _netMqGeneration;
    private string _netMqState = "never-started";

    private Mesh _cubeMesh;
    private MeshRenderer _meshRenderer;
    private Material _runtimeMaterial;
    private ComputeBuffer _matrixBuffer;
    private ComputeBuffer _colorBuffer;
    private ComputeBuffer _argsBuffer;
    private readonly uint[] _argsData = new uint[5];
    private Matrix4x4[] _matrixUpload;
    private Vector4[] _colorUpload;
    private int _lastRepackedPoints;
    private Bounds _renderBounds;
    private Matrix4x4 _renderPoseMatrix = Matrix4x4.identity;
    private MaterialPropertyBlock _mpb;
    private int _allocatedCapacity;
    private int _pointCount;
    private int _activeSourceCount;
    private int _bufferAllocationCount;
    private long _drawCount;
    private long _frameRxCount;
    private long _uploadCount;
    private long _lastUploadDurationTicks;
    private long _selectionStartTick;
    private long _identityLatencyTicks = -1;
    private long _firstPayloadLatencyTicks = -1;
    private long _firstUploadLatencyTicks = -1;
    private long _identityDropCount;
    private int _identityConfirmed;
    private long _lastSessionHeartbeatTick;
    private SessionStateHeartbeat _lastSessionHeartbeat;
    private float _nextHeartbeatAt;
    private int _combinedOverflowWarningCount;
    private bool _runtimeReady = true;
    private string _runtimeBlockReason;
    private bool _loggedRuntimeCapabilities;
    private bool _loggedFirstCombinedFrameDiagnostics;
    private bool _loggedFirstDrawDiagnostics;
    private bool _loggedFirstAnchoredDrawDiagnostics;
    private bool _loggedFirstPostAlignDrawDiagnostics;
    private bool _loggedMaterialStateDiagnostics;
    private bool _loggedUploadedMatrixDiagnostics;
    private bool _loggedWaitingForSceneAnchor;
    private bool _loggedAttachDiagnostics;
    private bool _loggedDrawStabilityWaiting;
    private bool _firstDrawUnlocked;
    private Transform _sceneRoot;
    private Transform _sceneAnchor;
    private string _sceneAnchorName;
    private float _nextRuntimeBlockedLogTime = float.NegativeInfinity;
    private QuestTrackingOriginGuard _trackingOriginGuard;
    private int _stableDrawChecks;
    private float _lastObservedDrawRecoveryAt = float.NegativeInfinity;
    [SerializeField, HideInInspector] private bool _hasExplicitRenderModeSelection;
    private bool _hasCachedOriginalMaterialState;
    private int _cachedRenderQueue;
    private float _cachedSrcBlend;
    private float _cachedDstBlend;
    private float _cachedZWrite;
    private float _cachedZTest;
    private Color _cachedBaseColor;
    private float _cachedOverlayAlpha;
    private float _cachedVisibilityGain;
    private float _cachedMinVisibleBrightness;

    private void Awake()
    {
        AsyncIO.ForceDotNet.Force();
        EnsureSourceStateMap();
        EnsureRendererResources();
        TryAttachToSceneRoot();
        UpdateRenderPose();
        _lastAnyPayloadTick = System.Diagnostics.Stopwatch.GetTimestamp();
        _selectionStartTick = SessionRegistry.SelectedAtStopwatchTick > 0
            ? SessionRegistry.SelectedAtStopwatchTick
            : _lastAnyPayloadTick;
    }

    private void Start()
    {
        EnsureSourceStateMap();
        StartNetMq();
    }

    private void Update()
    {
        EnsureSourceStateMap();
        TryAttachToSceneRoot();
        ApplyPendingFrames();
        ClearIdleSources();
        TryReconnectIfIdle();
        EmitHeartbeatIfDue();
    }

    // Pose + draw in LateUpdate so we read the final anchor transform after
    // all Update() callbacks (including SceneAnchorManager) have run, and
    // after the OVR late-latch pass has settled tracking-dependent transforms.
    private void LateUpdate()
    {
        UpdateRenderPose();
        ApplyRuntimeMaterialMode();
        DrawPointCloud();
    }

    private void EmitHeartbeatIfDue()
    {
        if (heartbeatLogIntervalSeconds <= 0f) return;
        if (Time.unscaledTime < _nextHeartbeatAt) return;
        _nextHeartbeatAt = Time.unscaledTime + heartbeatLogIntervalSeconds;

        long now = System.Diagnostics.Stopwatch.GetTimestamp();
        double lastPayloadAge = (now - Interlocked.Read(ref _lastAnyPayloadTick)) * TimestampToSeconds;
        double heartbeatAge = (now - Interlocked.Read(ref _lastSessionHeartbeatTick)) * TimestampToSeconds;
        double uploadMs = Interlocked.Read(ref _lastUploadDurationTicks) * TimestampToSeconds * 1000.0;
        double identityMs = TicksToMilliseconds(Interlocked.Read(ref _identityLatencyTicks));
        double firstPayloadMs = TicksToMilliseconds(Interlocked.Read(ref _firstPayloadLatencyTicks));
        double firstUploadMs = TicksToMilliseconds(Interlocked.Read(ref _firstUploadLatencyTicks));
        SessionStateHeartbeat heartbeat;
        lock (_frameLock) heartbeat = _lastSessionHeartbeat;
        Debug.Log(
            $"[GpuMergedPointCloudLoader] hb name='{name}' "
            + $"rx={Interlocked.Read(ref _frameRxCount)} draws={Interlocked.Read(ref _drawCount)} "
            + $"sources={_activeSourceCount}/{_sourceList.Count} points={_pointCount} "
            + $"sourceRx='{FormatSourceReceiveCounts()}' "
            + $"uploads={Interlocked.Read(ref _uploadCount)} uploadedPoints={_lastRepackedPoints} "
            + $"uploadMs={uploadMs:F2} identityMs={identityMs:F2} "
            + $"firstPayloadMs={firstPayloadMs:F2} firstUploadMs={firstUploadMs:F2} "
            + $"gc0={GC.CollectionCount(0)} "
            + $"firstDrawUnlocked={_firstDrawUnlocked} runtimeReady={_runtimeReady} "
            + $"sceneAnchor={(_sceneAnchor != null ? _sceneAnchor.name : "<null>")} "
            + $"lastPayloadAge_s={lastPayloadAge:F2} blockReason='{_runtimeBlockReason ?? ""}' "
            + $"identity={Interlocked.CompareExchange(ref _identityConfirmed, 0, 0)}/{expectedSessionIndex} "
            + $"identityDrops={Interlocked.Read(ref _identityDropCount)} sessionHbAge_s={heartbeatAge:F2} "
            + $"session={(heartbeat != null ? heartbeat.session_index : -1)} "
            + $"episode='{(heartbeat != null ? heartbeat.episode_id : "")}' "
            + $"policySeq={(heartbeat != null ? heartbeat.policy_seq : -1)} "
            + $"frame={(heartbeat != null ? heartbeat.frame_idx : -1)} "
            + $"mode='{(heartbeat != null ? heartbeat.mode : "")}' "
            + $"phase='{(heartbeat != null ? heartbeat.intervention_phase : "")}' "
            + $"upstreamStale={(heartbeat != null && heartbeat.upstream_stale)} "
            + $"upstreamStaleAge_s={(heartbeat != null ? heartbeat.upstream_stale_age_s : 0f):F2} "
            + $"qposDelta={(heartbeat != null ? heartbeat.qpos_delta_norm : 0f):F6} "
            + $"stateChangeAge_s={(heartbeat != null ? heartbeat.state_change_age_s : 0f):F2} "
            + $"endpoint=tcp://{publisherIp}:{topicPort} netmqGen={_netMqGeneration} netmqState='{_netMqState}'"
        );
    }

    private static double TicksToMilliseconds(long ticks)
    {
        return ticks < 0 ? -1.0 : ticks * TimestampToSeconds * 1000.0;
    }

    private void OnApplicationQuit()
    {
        _shuttingDown = true;
        ShutdownNetMq("application quit");
    }

    private void OnDisable()
    {
        // Stop the NetMQ poller while the shared context is still alive. Scene transitions disable
        // us before LoadScene so the poller is gone before the XR framework node's OnDestroy terminates
        // the context — otherwise the still-running poller throws TerminatingException and aborts
        // the app under IL2CPP. Idempotent with OnDestroy (sockets are lock-nulled).
        _shuttingDown = true;
        ShutdownNetMq("disable");
    }

    private void OnDestroy()
    {
        _shuttingDown = true;
        ShutdownNetMq("destroy");
        ReleaseBuffers();
        DestroyRuntimeMaterial();
    }

    public void ResetBootstrapBindingState()
    {
        long now = System.Diagnostics.Stopwatch.GetTimestamp();
        EnsureSourceStateMap();
        _sceneRoot = null;
        _sceneAnchor = null;
        _sceneAnchorName = string.Empty;
        _runtimeReady = true;
        _runtimeBlockReason = null;
        _nextRuntimeBlockedLogTime = float.NegativeInfinity;
        _loggedFirstCombinedFrameDiagnostics = false;
        _loggedFirstDrawDiagnostics = false;
        _loggedAttachDiagnostics = false;
        _loggedWaitingForSceneAnchor = false;
        _loggedFirstAnchoredDrawDiagnostics = false;
        _loggedFirstPostAlignDrawDiagnostics = false;
        _loggedMaterialStateDiagnostics = false;
        _loggedUploadedMatrixDiagnostics = false;
        _loggedDrawStabilityWaiting = false;
        _loggedRuntimeCapabilities = false;
        _combinedOverflowWarningCount = 0;
        _firstDrawUnlocked = false;
        _stableDrawChecks = 0;
        _lastObservedDrawRecoveryAt = float.NegativeInfinity;
        Interlocked.Exchange(ref _frameRxCount, 0);
        Interlocked.Exchange(ref _drawCount, 0);
        Interlocked.Exchange(ref _uploadCount, 0);
        Interlocked.Exchange(ref _lastUploadDurationTicks, 0);
        long selectionTick = SessionRegistry.SelectedAtStopwatchTick > 0
            ? SessionRegistry.SelectedAtStopwatchTick
            : now;
        Interlocked.Exchange(ref _selectionStartTick, selectionTick);
        Interlocked.Exchange(ref _identityLatencyTicks, -1);
        Interlocked.Exchange(ref _firstPayloadLatencyTicks, -1);
        Interlocked.Exchange(ref _firstUploadLatencyTicks, -1);
        Interlocked.Exchange(ref _lastAnyPayloadTick, now);
        Interlocked.Exchange(ref _identityConfirmed, expectedSessionIndex < 0 ? 1 : 0);
        Interlocked.Exchange(ref _identityDropCount, 0);
        Interlocked.Exchange(ref _lastSessionHeartbeatTick, now);
        lock (_frameLock)
        {
            _lastSessionHeartbeat = null;
            for (int i = 0; i < _sourceList.Count; i++)
            {
                var state = _sourceList[i];
                state.DecodeFrame = new DecodedFrame();
                state.PendingFrame = null;
                state.RenderFrame = null;
                state.LastPayloadTick = now;
                state.ContractWarningCount = 0;
                state.LoggedFirstFrameDiagnostics = false;
                state.LoggedIdleClear = false;
                Interlocked.Exchange(ref state.RxCount, 0);
            }
        }
        ClearCombinedDrawState();
        Debug.Log(
            $"[GpuMergedPointCloudLoader] Reset bootstrap binding state name='{name}' "
            + $"endpoint=tcp://{publisherIp}:{topicPort} topics={FormatTopicList()} "
            + $"netmqState='{_netMqState}' hasSub={_sub != null}"
        );
    }

    public void SetRenderMode(PointCloudRenderMode value)
    {
        renderMode = value;
        _hasExplicitRenderModeSelection = true;
        ApplyRuntimeMaterialMode();
    }

    private void EnsureSourceStateMap()
    {
        var normalizedTopics = new List<string>();
        if (topics != null)
        {
            for (int i = 0; i < topics.Length; i++)
            {
                string topic = topics[i];
                if (string.IsNullOrWhiteSpace(topic))
                    continue;
                if (normalizedTopics.Contains(topic))
                    continue;
                normalizedTopics.Add(topic);
            }
        }

        string signature = string.Join("|", normalizedTopics.ToArray());
        if (string.Equals(signature, _topicSignature, StringComparison.Ordinal))
            return;

        var preservedStates = new Dictionary<string, SourceState>(_sourceByTopic, StringComparer.Ordinal);
        _sourceByTopic.Clear();
        _sourceList.Clear();

        long now = System.Diagnostics.Stopwatch.GetTimestamp();
        for (int i = 0; i < normalizedTopics.Count; i++)
        {
            string topic = normalizedTopics[i];
            SourceState state;
            if (!preservedStates.TryGetValue(topic, out state) || state == null)
            {
                state = new SourceState();
                state.Topic = topic;
                state.Label = BuildSourceLabel(topic);
                state.LastPayloadTick = now;
            }
            else
            {
                state.Topic = topic;
                state.Label = BuildSourceLabel(topic);
            }

            _sourceByTopic[topic] = state;
            _sourceList.Add(state);
        }

        _topicSignature = signature;
    }

    private void EnsureRendererResources()
    {
        if (_meshRenderer == null)
            _meshRenderer = GetComponent<MeshRenderer>();

        if (_meshRenderer != null)
        {
            _meshRenderer.enabled = false;
            _meshRenderer.shadowCastingMode = ShadowCastingMode.Off;
            _meshRenderer.receiveShadows = false;
        }

        if (_cubeMesh == null)
        {
            var temp = GameObject.CreatePrimitive(PrimitiveType.Cube);
            _cubeMesh = temp.GetComponent<MeshFilter>().sharedMesh;
            if (Application.isPlaying)
                Destroy(temp);
            else
                DestroyImmediate(temp);
        }

        if (_renderBounds.size == Vector3.zero)
            _renderBounds = new Bounds(transform.position, Vector3.one * Mathf.Max(1.0f, renderBoundsSize));
        else
            _renderBounds.size = Vector3.one * Mathf.Max(1.0f, renderBoundsSize);

        ResolveInstancedMaterial();
        EvaluateRuntimeSupport();

        if (!_loggedRuntimeCapabilities)
        {
            _loggedRuntimeCapabilities = true;
            Debug.Log(
                "[GpuMergedPointCloudLoader] "
                + $"build='{LoaderBuildStamp}' name='{name}' api={SystemInfo.graphicsDeviceType} "
                + $"topics={FormatTopicList()} maxPointsPerSource={maxPointsPerSource} maxCombinedPoints={maxCombinedPoints} "
                + $"renderMode={GetEffectiveRenderMode()} supportsInstancing={SystemInfo.supportsInstancing} "
                + $"material={(instancedMaterial != null ? instancedMaterial.name : "null")} "
                + $"shader={(instancedMaterial != null && instancedMaterial.shader != null ? instancedMaterial.shader.name : "null")}"
            );
        }
    }

    private void ResolveInstancedMaterial()
    {
        Material sourceMaterial = instancedMaterial;
        if (sourceMaterial != null)
        {
            if (sourceMaterial.shader == null)
            {
                BlockRuntime("Assigned instanced material has no shader.");
                Debug.LogError($"[GpuMergedPointCloudLoader] {_runtimeBlockReason}");
                return;
            }

            EnsureRuntimeMaterialInstance(sourceMaterial);
            return;
        }

        if (_meshRenderer != null && _meshRenderer.sharedMaterial != null)
        {
            sourceMaterial = _meshRenderer.sharedMaterial;
            if (sourceMaterial.shader == null)
            {
                BlockRuntime("MeshRenderer shared material has no shader.");
                Debug.LogError($"[GpuMergedPointCloudLoader] {_runtimeBlockReason}");
                return;
            }

            EnsureRuntimeMaterialInstance(sourceMaterial);
            return;
        }

        if (_runtimeMaterial != null)
        {
            instancedMaterial = _runtimeMaterial;
            return;
        }

        Shader shader = Shader.Find("Custom/StableInstancedPoint");
        if (shader == null)
        {
            BlockRuntime("Shader 'Custom/StableInstancedPoint' could not be found.");
            Debug.LogError($"[GpuMergedPointCloudLoader] {_runtimeBlockReason}");
            return;
        }

        _runtimeMaterial = new Material(shader)
        {
            name = "CubeRenderingMaterial",
            hideFlags = HideFlags.DontSave
        };
        _runtimeMaterial.enableInstancing = true;
        _runtimeMaterial.SetColor(BaseColorId, fallbackBaseColor);
        instancedMaterial = _runtimeMaterial;
        CacheOriginalMaterialState(_runtimeMaterial);
        ApplyRuntimeMaterialMode();
        if (_meshRenderer != null)
            _meshRenderer.sharedMaterial = _runtimeMaterial;
    }

    private void EnsureRuntimeMaterialInstance(Material sourceMaterial)
    {
        if (sourceMaterial == null)
            return;

        if (ReferenceEquals(_runtimeMaterial, sourceMaterial))
        {
            instancedMaterial = _runtimeMaterial;
            if (_meshRenderer != null && _meshRenderer.sharedMaterial != _runtimeMaterial)
                _meshRenderer.sharedMaterial = _runtimeMaterial;
            return;
        }

        DestroyRuntimeMaterial();
        _runtimeMaterial = new Material(sourceMaterial)
        {
            name = sourceMaterial.name,
            hideFlags = HideFlags.DontSave
        };
        _runtimeMaterial.enableInstancing = true;
        instancedMaterial = _runtimeMaterial;
        CacheOriginalMaterialState(_runtimeMaterial);
        ApplyRuntimeMaterialMode();
        if (_meshRenderer != null)
            _meshRenderer.sharedMaterial = _runtimeMaterial;
    }

    private void DestroyRuntimeMaterial()
    {
        if (_runtimeMaterial == null)
            return;

        if (Application.isPlaying)
            Destroy(_runtimeMaterial);
        else
            DestroyImmediate(_runtimeMaterial);
        _runtimeMaterial = null;
        _hasCachedOriginalMaterialState = false;
    }

    private bool BindMaterialResources()
    {
        if (instancedMaterial == null || instancedMaterial.shader == null)
            return false;

        instancedMaterial.enableInstancing = true;
        if (_matrixBuffer != null)
            instancedMaterial.SetBuffer(MatrixBufferId, _matrixBuffer);
        if (_colorBuffer != null)
            instancedMaterial.SetBuffer(ColorBufferId, _colorBuffer);
        // _RenderPoseMatrix is now set via _mpb in UpdateRenderPose (LateUpdate).
        return true;
    }

    private void EvaluateRuntimeSupport()
    {
        if (!_runtimeReady)
            return;

        if (instancedMaterial == null)
        {
            BlockRuntime("No instanced material is available for merged GPU_Merged point-cloud rendering.");
            return;
        }

        if (!SystemInfo.supportsInstancing)
            BlockRuntime("Merged GPU_Merged point-cloud rendering requires instancing support.");
    }

    private void EnsureCapacity(int requiredCapacity)
    {
        if (!_runtimeReady)
            return;

        requiredCapacity = Mathf.Max(1, requiredCapacity);
        if (_matrixBuffer != null
            && _colorBuffer != null
            && _argsBuffer != null
            && _allocatedCapacity >= requiredCapacity)
        {
            return;
        }

        ReleaseBuffers();
        _allocatedCapacity = requiredCapacity;
        _matrixUpload = new Matrix4x4[_allocatedCapacity];
        _colorUpload = new Vector4[_allocatedCapacity];
        _matrixBuffer = new ComputeBuffer(_allocatedCapacity, sizeof(float) * 16);
        _colorBuffer = new ComputeBuffer(_allocatedCapacity, sizeof(float) * 4);
        _argsBuffer = new ComputeBuffer(1, _argsData.Length * sizeof(uint), ComputeBufferType.IndirectArguments);
        _bufferAllocationCount++;

        if (logBufferLifecycle)
        {
            Debug.Log(
                $"[GpuMergedPointCloudLoader] Allocated buffers for '{name}' capacity={_allocatedCapacity} allocationCount={_bufferAllocationCount}."
            );
        }
    }

    private void ReleaseBuffers()
    {
        try { _matrixBuffer?.Release(); } catch { }
        try { _colorBuffer?.Release(); } catch { }
        try { _argsBuffer?.Release(); } catch { }
        _matrixBuffer = null;
        _colorBuffer = null;
        _argsBuffer = null;
        _matrixUpload = null;
        _colorUpload = null;
        _allocatedCapacity = 0;
    }

    private void OnMsg(object sender, NetMQSocketEventArgs e)
    {
        if (_shuttingDown)
            return;

        _netMessage.Clear();
        var msg = _netMessage;
        // Receiving can throw if the shared NetMQ context is terminated underneath us during
        // scene unload (the XR framework node's OnDestroy terminates the context, and OnDestroy order
        // is nondeterministic). An unhandled exception here propagates on the poller thread and
        // aborts the whole app under IL2CPP. Catch it, stop polling, and return cleanly.
        bool got;
        try
        {
            got = e.Socket.TryReceiveMultipartMessage(ref msg);
        }
        catch (TerminatingException)
        {
            _shuttingDown = true;
            try { e.Socket.ReceiveReady -= OnMsg; } catch { }
            return;
        }
        catch (ObjectDisposedException)
        {
            _shuttingDown = true;
            try { e.Socket.ReceiveReady -= OnMsg; } catch { }
            return;
        }
        if (!got || msg.FrameCount < 2)
            return;

        string receivedTopic;
        try { receivedTopic = msg[0].ConvertToString(); }
        catch { return; }

        ReadOnlySpan<byte> payload = msg[msg.FrameCount - 1].AsSpan();
        if (string.Equals(receivedTopic, SessionStateTopic, StringComparison.Ordinal))
        {
            AcceptSessionHeartbeat(payload);
            return;
        }

        SourceState state;
        if (!_sourceByTopic.TryGetValue(receivedTopic, out state) || state == null)
            return;

        if (expectedSessionIndex >= 0
            && Interlocked.CompareExchange(ref _identityConfirmed, 0, 0) == 0)
        {
            Interlocked.Increment(ref _identityDropCount);
            return;
        }

        Interlocked.Increment(ref _frameRxCount);
        Interlocked.Increment(ref state.RxCount);
        if (!GpuPointCloudContract.TryParseHeader(payload, out var header, out var error))
        {
            LogContractWarning(state, error);
            return;
        }

        if (header.DeclaredCapacity > maxPointsPerSource)
        {
            LogContractWarning(
                state,
                $"Dropped '{receivedTopic}' because declared capacity {header.DeclaredCapacity} exceeds maxPointsPerSource {maxPointsPerSource}. "
                + "Raise maxPointsPerSource (InterventionSessionBootstrap) or lower the publisher's --pc_max_points."
            );
            // The source IS alive — we are rejecting its payload, not missing it. Without
            // this the idle timer kept running, the source was cleared, and since every
            // later payload is rejected for the same reason it could never come back: a
            // permanent silent loss of one camera that looks exactly like a network drop.
            Interlocked.Exchange(ref state.LastPayloadTick, System.Diagnostics.Stopwatch.GetTimestamp());
            return;
        }

        var frame = state.DecodeFrame ?? new DecodedFrame();
        // MUST size from DeclaredCapacity: GpuPointCloudContract.TryDecodeInto rejects any
        // destination buffer shorter than the declared capacity ("Raw point buffer is
        // missing or smaller than the declared capacity") and returns false, which drops the
        // payload entirely. Sizing this from ActualCount instead made EVERY payload fail
        // that check and the cloud disappeared completely.
        //
        // The publisher advertises an adaptive declared capacity, so the known-good
        // latest-frame handoff does not reserve the configured maximum for every frame.
        frame.EnsureCapacity(Mathf.Max(1, header.DeclaredCapacity));
        if (!GpuPointCloudContract.TryDecodeInto(
                payload,
                header,
                frame.RawPoints,
                frame.Colors,
                positionScale,
                false,
                fallbackBaseColor,
                out error))
        {
            LogContractWarning(state, error);
            return;
        }

        if (Interlocked.Read(ref _firstPayloadLatencyTicks) < 0)
        {
            long latency = System.Diagnostics.Stopwatch.GetTimestamp()
                - Interlocked.Read(ref _selectionStartTick);
            Interlocked.CompareExchange(ref _firstPayloadLatencyTicks, latency, -1);
        }

        frame.DeclaredCapacity = header.DeclaredCapacity;
        frame.ActualCount = header.ActualCount;

        long now = System.Diagnostics.Stopwatch.GetTimestamp();
        Interlocked.Exchange(ref state.LastPayloadTick, now);
        Interlocked.Exchange(ref _lastAnyPayloadTick, now);
        state.LoggedIdleClear = false;

        lock (_frameLock)
        {
            var previousPending = state.PendingFrame;
            state.PendingFrame = frame;
            state.DecodeFrame = previousPending != null && !ReferenceEquals(previousPending, frame)
                ? previousPending
                : new DecodedFrame();
        }
    }

    private void AcceptSessionHeartbeat(ReadOnlySpan<byte> payload)
    {
        SessionStateHeartbeat heartbeat;
        try
        {
            heartbeat = JsonUtility.FromJson<SessionStateHeartbeat>(Encoding.UTF8.GetString(payload));
        }
        catch
        {
            return;
        }
        if (heartbeat == null)
            return;

        Interlocked.Exchange(ref _lastSessionHeartbeatTick, System.Diagnostics.Stopwatch.GetTimestamp());
        lock (_frameLock) _lastSessionHeartbeat = heartbeat;

        bool matches = (expectedSessionIndex < 0 || heartbeat.session_index == expectedSessionIndex)
            && heartbeat.topic_port == topicPort;
        if (!matches)
            return;

        if (Interlocked.Exchange(ref _identityConfirmed, 1) == 0)
        {
            long latency = System.Diagnostics.Stopwatch.GetTimestamp()
                - Interlocked.Read(ref _selectionStartTick);
            Interlocked.CompareExchange(ref _identityLatencyTicks, latency, -1);
            Debug.Log(
                $"[GpuMergedPointCloudLoader] session_state accepted "
                + $"expectedSession={expectedSessionIndex} session={heartbeat.session_index} "
                + $"port={heartbeat.topic_port} episode='{heartbeat.episode_id}' "
                + $"policySeq={heartbeat.policy_seq} frame={heartbeat.frame_idx} "
                + $"mode='{heartbeat.mode}' phase='{heartbeat.intervention_phase}'."
            );
        }
    }

    private string FormatSourceReceiveCounts()
    {
        var parts = new string[_sourceList.Count];
        for (int i = 0; i < _sourceList.Count; i++)
        {
            SourceState state = _sourceList[i];
            parts[i] = $"{state.Label}:{Interlocked.Read(ref state.RxCount)}";
        }
        return string.Join(",", parts);
    }

    private void ApplyPendingFrames()
    {
        bool changed = false;
        lock (_frameLock)
        {
            for (int i = 0; i < _sourceList.Count; i++)
            {
                var state = _sourceList[i];
                if (state.PendingFrame == null)
                    continue;

                state.RenderFrame = state.PendingFrame;
                state.PendingFrame = null;
                changed = true;
            }
        }

        if (!changed)
            return;

        LogSourceFrameDiagnostics();
        RebuildCombinedCloud();
    }

    private void RebuildCombinedCloud()
    {
        if (!_runtimeReady)
        {
            ClearCombinedDrawState();
            LogRuntimeBlockedThrottled("Skipping merged point-cloud upload");
            return;
        }

        int safeCombinedCapacity = Mathf.Max(1, maxCombinedPoints);
        EnsureCapacity(safeCombinedCapacity);
        if (!HasUploadResources(safeCombinedCapacity))
        {
            BlockRuntime("Merged GPU_Merged upload resources are unavailable.");
            ClearCombinedDrawState();
            LogRuntimeBlockedThrottled("Skipping merged point-cloud upload");
            return;
        }

        long uploadStarted = System.Diagnostics.Stopwatch.GetTimestamp();
        int writeIndex = 0;
        int activeSources = 0;
        float safeCubeSize = Mathf.Max(0.0001f, ResolveEffectiveCubeSize() * scale);
        var sourceSummary = _loggedFirstCombinedFrameDiagnostics ? null : new StringBuilder(128);
        for (int sourceIndex = 0; sourceIndex < _sourceList.Count; sourceIndex++)
        {
            var state = _sourceList[sourceIndex];
            var frame = state.RenderFrame;
            if (frame == null || frame.RawPoints == null || frame.Colors == null || frame.ActualCount <= 0)
                continue;

            activeSources++;
            if (sourceSummary != null)
            {
                if (sourceSummary.Length > 0)
                    sourceSummary.Append(' ');
                sourceSummary.Append(state.Label).Append(':').Append(frame.ActualCount);
            }

            int safeCount = Mathf.Min(frame.ActualCount, Mathf.Min(frame.RawPoints.Length, frame.Colors.Length));
            for (int i = 0; i < safeCount; i++)
            {
                if (writeIndex >= safeCombinedCapacity)
                {
                    if (_combinedOverflowWarningCount < 10)
                        Debug.LogWarning($"[GpuMergedPointCloudLoader] Combined point budget {safeCombinedCapacity} reached on '{name}'. Remaining source data will be clipped.");
                    _combinedOverflowWarningCount++;
                    break;
                }
                Vector3 point = frame.RawPoints[i] + debugOffset;
                _matrixUpload[writeIndex] = Matrix4x4.TRS(
                    point * scale, Quaternion.identity, Vector3.one * safeCubeSize);

                Color32 color = frame.Colors[i];
                _colorUpload[writeIndex] = new Vector4(
                    color.r / 255.0f,
                    color.g / 255.0f,
                    color.b / 255.0f,
                    1.0f
                );
                writeIndex++;
            }

            if (writeIndex >= safeCombinedCapacity)
                break;
        }

        _activeSourceCount = activeSources;
        _lastRepackedPoints = writeIndex;
        if (writeIndex <= 0)
        {
            ClearCombinedDrawState();
            return;
        }
        _matrixBuffer.SetData(_matrixUpload, 0, 0, writeIndex);
        _colorBuffer.SetData(_colorUpload, 0, 0, writeIndex);
        Interlocked.Exchange(
            ref _lastUploadDurationTicks,
            System.Diagnostics.Stopwatch.GetTimestamp() - uploadStarted);
        Interlocked.Increment(ref _uploadCount);
        if (Interlocked.Read(ref _firstUploadLatencyTicks) < 0)
        {
            long latency = System.Diagnostics.Stopwatch.GetTimestamp()
                - Interlocked.Read(ref _selectionStartTick);
            Interlocked.CompareExchange(ref _firstUploadLatencyTicks, latency, -1);
        }
        if (!BindMaterialResources())
        {
            BlockRuntime("Merged GPU_Merged material bindings are unavailable.");
            ClearCombinedDrawState();
            LogRuntimeBlockedThrottled("Skipping merged point-cloud upload");
            return;
        }

        _pointCount = writeIndex;
        if (!_loggedFirstCombinedFrameDiagnostics)
        {
            _loggedFirstCombinedFrameDiagnostics = true;
            Debug.Log(
                "[GpuMergedPointCloudLoader] First combined frame "
                + $"build='{LoaderBuildStamp}' name='{name}' activeSources={_activeSourceCount}/{_sourceList.Count} "
                + $"totalPoints={_pointCount} maxCombinedPoints={maxCombinedPoints} sources={sourceSummary}"
            );
        }
    }

    private void DrawPointCloud()
    {
        if (!_runtimeReady)
        {
            if (_pointCount > 0)
                LogRuntimeBlockedThrottled("Skipping merged point-cloud draw");
            return;
        }

        if (!renderEnabled || _pointCount <= 0)
            return;

        if (!IsReadyForStableAnchoredDraw())
            return;

        if (!HasDrawResources())
        {
            BlockRuntime("Merged GPU_Merged draw resources are unavailable.");
            ClearCombinedDrawState();
            LogRuntimeBlockedThrottled("Skipping merged point-cloud draw");
            return;
        }

        float effectiveCubeSize = Mathf.Max(0.0001f, ResolveEffectiveCubeSize() * scale);
        _renderBounds.center = ExtractTranslation(_renderPoseMatrix);
        _renderBounds.size = Vector3.one * Mathf.Max(1.0f, renderBoundsSize);
        if (!BindMaterialResources())
        {
            BlockRuntime("Merged GPU_Merged draw material bindings are unavailable.");
            ClearCombinedDrawState();
            LogRuntimeBlockedThrottled("Skipping merged point-cloud draw");
            return;
        }

        _argsData[0] = _cubeMesh.GetIndexCount(0);
        _argsData[1] = (uint)_pointCount;
        _argsData[2] = _cubeMesh.GetIndexStart(0);
        _argsData[3] = _cubeMesh.GetBaseVertex(0);
        _argsData[4] = 0u;
        _argsBuffer.SetData(_argsData);

        Graphics.DrawMeshInstancedIndirect(
            _cubeMesh,
            0,
            instancedMaterial,
            _renderBounds,
            _argsBuffer,
            0,
            _mpb,
            ShadowCastingMode.Off,
            false,
            gameObject.layer
        );
        Interlocked.Increment(ref _drawCount);

        if (!_loggedFirstDrawDiagnostics)
        {
            _loggedFirstDrawDiagnostics = true;
            Debug.Log(
                "[GpuMergedPointCloudLoader] First combined draw "
                + $"name='{name}' renderMode={GetEffectiveRenderMode()} points={_pointCount} activeSources={_activeSourceCount}/{_sourceList.Count} "
                + $"boundsCenter={_renderBounds.center} boundsSize={_renderBounds.size} poseMatrix={FormatMatrix(_renderPoseMatrix)}"
            );
        }

        if (_sceneAnchor != null && !_loggedFirstAnchoredDrawDiagnostics)
        {
            _loggedFirstAnchoredDrawDiagnostics = true;
            Debug.Log(
                "[GpuMergedPointCloudLoader] First anchored combined draw "
                + $"name='{name}' renderMode={GetEffectiveRenderMode()} anchorPath='{GetTransformPath(_sceneAnchor)}' "
                + $"points={_pointCount} activeSources={_activeSourceCount}/{_sourceList.Count} poseMatrix={FormatMatrix(_renderPoseMatrix)}"
            );
        }

        if (!_loggedMaterialStateDiagnostics)
        {
            _loggedMaterialStateDiagnostics = true;
            Debug.Log(
                "[GpuMergedPointCloudLoader] Material state "
                + $"name='{name}' renderMode={GetEffectiveRenderMode()} material='{(instancedMaterial != null ? instancedMaterial.name : "<none>")}' "
                + $"shader='{(instancedMaterial != null && instancedMaterial.shader != null ? instancedMaterial.shader.name : "<none>")}' "
                + $"renderQueue={(instancedMaterial != null ? instancedMaterial.renderQueue : -1)} "
                + $"zWrite={FormatMaterialFlag(GetMaterialFloat(instancedMaterial, "_ZWrite", 0.0f))} "
                + $"zTest={FormatCompareFunction(GetMaterialFloat(instancedMaterial, "_ZTest", 4.0f))} "
                + $"blend={FormatBlendMode(GetMaterialFloat(instancedMaterial, "_SrcBlend", 1.0f))}/{FormatBlendMode(GetMaterialFloat(instancedMaterial, "_DstBlend", 0.0f))} "
                + $"cubeSize={effectiveCubeSize:F4} pointCount={_pointCount} activeSources={_activeSourceCount}/{_sourceList.Count}"
            );
        }

        if (!_loggedFirstPostAlignDrawDiagnostics && ShouldLogPostAlignDrawDiagnostics())
            LogPostAlignDrawDiagnostics(effectiveCubeSize);
    }

    private void ClearIdleSources()
    {
        bool changed = false;
        long now = System.Diagnostics.Stopwatch.GetTimestamp();
        for (int i = 0; i < _sourceList.Count; i++)
        {
            var state = _sourceList[i];
            if (state.RenderFrame == null)
                continue;

            double idleFor = (now - Interlocked.Read(ref state.LastPayloadTick)) * TimestampToSeconds;
            if (idleFor < Mathf.Max(0.25f, sourceIdleClearSeconds))
                continue;

            state.RenderFrame = null;
            changed = true;
            if (!state.LoggedIdleClear)
            {
                state.LoggedIdleClear = true;
                Debug.LogWarning(
                    $"[GpuMergedPointCloudLoader] Cleared idle source '{state.Label}' on '{name}' after {idleFor:F2}s without payloads."
                );
            }
        }

        if (changed)
            RebuildCombinedCloud();
    }

    private void TryReconnectIfIdle()
    {
        if (!autoReconnectOnIdle || Time.unscaledTime < _nextReconnectTime)
            return;

        double idleFor =
            (System.Diagnostics.Stopwatch.GetTimestamp() - Interlocked.Read(ref _lastAnyPayloadTick))
            * TimestampToSeconds;
        if (idleFor < Mathf.Max(0.5f, reconnectIdleSeconds))
            return;

        if (_pointCount > 0)
        {
            Debug.LogWarning(
                $"[GpuMergedPointCloudLoader] Clearing merged cloud on '{name}' after {idleFor:F2}s idle across all topics."
            );
            ClearCombinedDrawState();
        }

        Debug.LogWarning(
            $"[GpuMergedPointCloudLoader] Idle for {idleFor:F2}s on '{name}'. "
            + $"Keeping existing subscriber alive; topics={FormatTopicList()}."
        );
        _nextReconnectTime = Time.unscaledTime + Mathf.Max(0.5f, reconnectCooldownSeconds);
    }

    private void StartNetMq()
    {
        if (_shuttingDown)
        {
            Debug.Log(
                $"[GpuMergedPointCloudLoader] StartNetMq skipped for '{name}' because loader is shutting down."
            );
            return;
        }

        SubscriberSocket sub = null;
        NetMQPoller poller = null;
        int generation = ++_netMqGeneration;
        _netMqState = "starting";
        Debug.Log(
            $"[GpuMergedPointCloudLoader] NetMQ start gen={generation} name='{name}' "
            + $"endpoint=tcp://{publisherIp}:{topicPort} topics={FormatTopicList()}"
        );
        try
        {
            sub = new SubscriberSocket();
            sub.Options.ReceiveHighWatermark = 2;
            sub.Options.Linger = TimeSpan.Zero;
            sub.Connect($"tcp://{publisherIp}:{topicPort}");
            for (int i = 0; i < _sourceList.Count; i++)
                sub.Subscribe(_sourceList[i].Topic);
            sub.Subscribe(SessionStateTopic);
            sub.ReceiveReady += OnMsg;

            poller = new NetMQPoller { sub };
            lock (_netMqLock)
            {
                _sub = sub;
                _poller = poller;
            }

            poller.RunAsync();
            Interlocked.Exchange(ref _lastAnyPayloadTick, System.Diagnostics.Stopwatch.GetTimestamp());
            _nextReconnectTime = Time.unscaledTime + Mathf.Max(0.5f, reconnectCooldownSeconds);
            _netMqState = "connected";
            Debug.Log(
                $"[GpuMergedPointCloudLoader] Connected gen={generation} tcp://{publisherIp}:{topicPort} topics={FormatTopicList()} for '{name}'."
            );
        }
        catch (Exception ex)
        {
            lock (_netMqLock)
            {
                if (ReferenceEquals(_sub, sub))
                    _sub = null;
                if (ReferenceEquals(_poller, poller))
                    _poller = null;
            }

            try { if (sub != null) sub.ReceiveReady -= OnMsg; } catch { }
            try { if (poller != null && poller.IsRunning) poller.Stop(); } catch { }
            try { sub?.Close(); } catch { }
            try { sub?.Dispose(); } catch { }
            try { poller?.Dispose(); } catch { }

            long staleTick = System.Diagnostics.Stopwatch.GetTimestamp()
                - (long)(Mathf.Max(0.5f, reconnectIdleSeconds) * System.Diagnostics.Stopwatch.Frequency);
            Interlocked.Exchange(ref _lastAnyPayloadTick, staleTick);
            _nextReconnectTime = Time.unscaledTime + Mathf.Max(0.25f, reconnectCooldownSeconds);
            _netMqState = "start-failed";
            Debug.LogWarning(
                $"[GpuMergedPointCloudLoader] NetMQ start failed gen={generation} for '{name}' "
                + $"tcp://{publisherIp}:{topicPort}: {ex.Message}."
            );
        }
    }

    /// <summary>Re-point this loader at a different session and reconnect.
    ///
    /// In combined view the grid and single view are both loaded, and switching sessions
    /// happens in place (no scene reload). Without this, the loader kept its socket on the
    /// PREVIOUSLY selected session — which both wasted bandwidth and inflated that
    /// session's ZMQ peer count, causing Python to keep it in ACTIVE (full point-cloud)
    /// mode long after it was deselected.</summary>
    public void Reconfigure(string ip, int newTopicPort, int newSessionIndex = -1)
    {
        bool endpointChanged = publisherIp != ip || topicPort != newTopicPort;
        bool identityChanged = expectedSessionIndex != newSessionIndex;
        if (!endpointChanged && !identityChanged)
            return;
        publisherIp = ip;
        topicPort = newTopicPort;
        expectedSessionIndex = newSessionIndex;
        Interlocked.Exchange(ref _identityConfirmed, expectedSessionIndex < 0 ? 1 : 0);
        Interlocked.Exchange(ref _identityDropCount, 0);
        long selectionTick = SessionRegistry.SelectedAtStopwatchTick > 0
            ? SessionRegistry.SelectedAtStopwatchTick
            : System.Diagnostics.Stopwatch.GetTimestamp();
        Interlocked.Exchange(ref _lastSessionHeartbeatTick, selectionTick);
        Interlocked.Exchange(ref _selectionStartTick, selectionTick);
        Interlocked.Exchange(ref _identityLatencyTicks, -1);
        Interlocked.Exchange(ref _firstPayloadLatencyTicks, -1);
        Interlocked.Exchange(ref _firstUploadLatencyTicks, -1);
        // Drop the previous session's frames only. Deliberately NOT
        // ResetBootstrapBindingState(): that also clears _sceneRoot/_sceneAnchor, which
        // would re-anchor the cloud and produce a visible jump on every session switch.
        lock (_frameLock)
        {
            _lastSessionHeartbeat = null;
            long now = System.Diagnostics.Stopwatch.GetTimestamp();
            for (int i = 0; i < _sourceList.Count; i++)
            {
                var state = _sourceList[i];
                state.PendingFrame = null;
                state.RenderFrame = null;
                state.LastPayloadTick = now;
                state.LoggedIdleClear = false;
            }
        }
        ClearCombinedDrawState();
        if (endpointChanged)
            ReconnectNetMq("session switch");
    }

    private void ReconnectNetMq(string reason)
    {
        Debug.Log(
            $"[GpuMergedPointCloudLoader] Reconnect requested for '{name}' reason='{reason}' "
            + $"rx={Interlocked.Read(ref _frameRxCount)} draws={Interlocked.Read(ref _drawCount)} "
            + $"sources={_activeSourceCount}/{_sourceList.Count} endpoint=tcp://{publisherIp}:{topicPort}"
        );
        ShutdownNetMq($"reconnect: {reason}");
        _shuttingDown = false;
        StartNetMq();
    }

    private void ShutdownNetMq(string reason)
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

        _netMqState = "stopping";
        Debug.Log(
            $"[GpuMergedPointCloudLoader] NetMQ shutdown reason='{reason}' name='{name}' "
            + $"hadSub={sub != null} hadPoller={poller != null} pollerRunning={(poller != null && poller.IsRunning)} "
            + $"rx={Interlocked.Read(ref _frameRxCount)} draws={Interlocked.Read(ref _drawCount)} "
            + $"sources={_activeSourceCount}/{_sourceList.Count} endpoint=tcp://{publisherIp}:{topicPort}"
        );
        try { if (sub != null) sub.ReceiveReady -= OnMsg; } catch { }
        // Stop() BLOCKS until the poll loop exits, so the socket is disposed only after the
        // poller stops iterating it. StopAsync()+immediate Dispose raced → the running poller
        // called Remove() on a disposed socket → "Must not be disposed" → crash on scene exit.
        try { if (poller != null && poller.IsRunning) poller.Stop(); } catch { }
        try { sub?.Close(); } catch { }
        try { sub?.Dispose(); } catch { }
        try { poller?.Dispose(); } catch { }
        _netMqState = "stopped";
    }

    private void TryAttachToSceneRoot()
    {
        if (!parentToSimScene)
        {
            _sceneRoot = null;
            _sceneAnchor = null;
            return;
        }

        Transform root = ResolveSceneRoot();
        if (root == null)
        {
            if (!_loggedWaitingForSceneAnchor)
            {
                _loggedWaitingForSceneAnchor = true;
                Debug.LogWarning($"[GpuMergedPointCloudLoader] Waiting for SimScene '{simSceneName}' before attaching '{name}'.");
            }
            return;
        }

        Transform anchor = ResolveOrCreateSceneAnchor(root);
        if (anchor == null)
            return;

        bool changed = root != _sceneRoot || anchor != _sceneAnchor;
        _sceneRoot = root;
        _sceneAnchor = anchor;
        _loggedWaitingForSceneAnchor = false;

        if (changed)
        {
            _loggedAttachDiagnostics = false;
            _loggedFirstAnchoredDrawDiagnostics = false;
            _loggedFirstPostAlignDrawDiagnostics = false;
            _loggedUploadedMatrixDiagnostics = false;
            _loggedDrawStabilityWaiting = false;
            _firstDrawUnlocked = false;
            _stableDrawChecks = 0;
        }

        if (logAttachInfo && !_loggedAttachDiagnostics)
        {
            _loggedAttachDiagnostics = true;
            Debug.Log(
                $"[GpuMergedPointCloudLoader] Bound '{name}' to SimScene '{root.name}' via anchor '{anchor.name}' path='{GetTransformPath(anchor)}'."
            );
        }
    }

    private void UpdateRenderPose()
    {
        // Read the scene root directly rather than through the _sceneAnchor child.
        // The child's local transform is always reset to identity in TryAttachToSceneRoot,
        // so both paths give the same matrix — but reading the root bypasses any
        // child-level interference from framework code.
        if (parentToSimScene && _sceneRoot != null)
            _renderPoseMatrix = _sceneRoot.localToWorldMatrix;
        else
            _renderPoseMatrix = transform.localToWorldMatrix;

        _renderBounds.center = ExtractTranslation(_renderPoseMatrix);
        _renderBounds.size = Vector3.one * Mathf.Max(1.0f, renderBoundsSize);
        // Write to a per-draw MaterialPropertyBlock instead of the shared material.
        // This is the Unity-recommended path for DrawMeshInstancedIndirect and
        // isolates the per-frame volatile matrix from static material state.
        if (_mpb == null) _mpb = new MaterialPropertyBlock();
        _mpb.SetMatrix(RenderPoseMatrixId, _renderPoseMatrix);
    }

    private float ResolveEffectiveCubeSize()
    {
        return debugCubeSizeOverride > 0.0f ? debugCubeSizeOverride : CubeSize;
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
        if (!string.IsNullOrWhiteSpace(sceneAnchorNameOverride))
            return sceneAnchorNameOverride;

        if (!string.IsNullOrEmpty(_sceneAnchorName))
            return _sceneAnchorName;

        string fragment = string.IsNullOrWhiteSpace(name) ? "combined" : name;
        var sanitized = new StringBuilder(fragment.Length);
        for (int i = 0; i < fragment.Length; i++)
        {
            char c = fragment[i];
            sanitized.Append(char.IsLetterOrDigit(c) ? c : '_');
        }

        _sceneAnchorName = $"PointCloudAnchor_{sanitized}";
        return _sceneAnchorName;
    }

    private QuestTrackingOriginGuard ResolveTrackingOriginGuard()
    {
        if (_trackingOriginGuard == null)
            _trackingOriginGuard = QuestTrackingOriginGuard.ResolveSharedGuard();
        return _trackingOriginGuard;
    }

    private bool IsReadyForStableAnchoredDraw()
    {
        var trackingOriginGuard = ResolveTrackingOriginGuard();
        if (trackingOriginGuard != null
            && trackingOriginGuard.LastRecoveryAt > _lastObservedDrawRecoveryAt)
        {
            _lastObservedDrawRecoveryAt = trackingOriginGuard.LastRecoveryAt;
            _firstDrawUnlocked = false;
            _stableDrawChecks = 0;
            _loggedDrawStabilityWaiting = false;
            _loggedFirstAnchoredDrawDiagnostics = false;
            Debug.LogWarning(
                $"[GpuMergedPointCloudLoader] XR recovery detected for '{name}'. Waiting for stable merged draw before resuming point-cloud rendering."
            );
        }

        if (_firstDrawUnlocked)
            return true;

        bool anchorResolved = !waitForSceneAnchorBeforeFirstDraw || _sceneAnchor != null;
        bool runtimeStable = !requireStableTrackingBeforeFirstDraw
            || trackingOriginGuard == null
            || trackingOriginGuard.IsRuntimeStable;
        bool worldFrameStable = !requireStableTrackingBeforeFirstDraw
            || trackingOriginGuard == null
            || trackingOriginGuard.IsWorldFrameStable;
        if (anchorResolved && runtimeStable && worldFrameStable)
            _stableDrawChecks = Mathf.Min(_stableDrawChecks + 1, Mathf.Max(1, stableChecksBeforeFirstDraw));
        else
            _stableDrawChecks = 0;

        if (_stableDrawChecks >= Mathf.Max(1, stableChecksBeforeFirstDraw))
        {
            _firstDrawUnlocked = true;
            _loggedDrawStabilityWaiting = false;
            Debug.Log(
                "[GpuMergedPointCloudLoader] Draw gate released "
                + $"name='{name}' anchorPath='{(_sceneAnchor != null ? GetTransformPath(_sceneAnchor) : "<none>")}' "
                + $"stableChecks={_stableDrawChecks}/{Mathf.Max(1, stableChecksBeforeFirstDraw)} "
                + $"worldFrameStable={worldFrameStable} "
                + $"rigRootDelta={(trackingOriginGuard != null ? trackingOriginGuard.CurrentRigRootWorldDelta.ToString("F3") : "n/a")} "
                + $"trackingSpaceDelta={(trackingOriginGuard != null ? trackingOriginGuard.CurrentTrackingSpaceWorldDelta.ToString("F3") : "n/a")}"
            );
            return true;
        }

        if (!_loggedDrawStabilityWaiting)
        {
            Debug.LogWarning(
                "[GpuMergedPointCloudLoader] Waiting for stable anchored draw "
                + $"name='{name}' anchorResolved={anchorResolved} runtimeStable={runtimeStable} "
                + $"worldFrameStable={worldFrameStable} hasWorldFrameBaseline={(trackingOriginGuard == null || trackingOriginGuard.HasWorldFrameBaseline)} "
                + $"stableChecks={_stableDrawChecks}/{Mathf.Max(1, stableChecksBeforeFirstDraw)} "
                + $"rigRootDelta={(trackingOriginGuard != null ? trackingOriginGuard.CurrentRigRootWorldDelta.ToString("F3") : "n/a")} "
                + $"trackingSpaceDelta={(trackingOriginGuard != null ? trackingOriginGuard.CurrentTrackingSpaceWorldDelta.ToString("F3") : "n/a")}"
            );
            _loggedDrawStabilityWaiting = true;
        }

        return false;
    }

    private void CacheOriginalMaterialState(Material material)
    {
        if (material == null)
            return;

        _cachedRenderQueue = material.renderQueue;
        _cachedSrcBlend = GetMaterialFloat(material, SrcBlendId, 5.0f);
        _cachedDstBlend = GetMaterialFloat(material, DstBlendId, 10.0f);
        _cachedZWrite = GetMaterialFloat(material, ZWriteId, 0.0f);
        _cachedZTest = GetMaterialFloat(material, ZTestId, 8.0f);
        _cachedBaseColor = GetMaterialColor(material, BaseColorId, fallbackBaseColor);
        _cachedOverlayAlpha = GetMaterialFloat(material, OverlayAlphaId, 0.95f);
        _cachedVisibilityGain = GetMaterialFloat(material, VisibilityGainId, 1.75f);
        _cachedMinVisibleBrightness = GetMaterialFloat(material, MinVisibleBrightnessId, 0.28f);
        _hasCachedOriginalMaterialState = true;
    }

    private void ApplyRuntimeMaterialMode()
    {
        Material material = instancedMaterial;
        if (material == null || material.shader == null || !_hasCachedOriginalMaterialState)
            return;

        if (GetEffectiveRenderMode() == PointCloudRenderMode.OverlayTransparent)
        {
            material.renderQueue = _cachedRenderQueue;
            SetMaterialFloat(material, SrcBlendId, _cachedSrcBlend);
            SetMaterialFloat(material, DstBlendId, _cachedDstBlend);
            SetMaterialFloat(material, ZWriteId, _cachedZWrite);
            SetMaterialFloat(material, ZTestId, _cachedZTest);
            SetMaterialColor(material, BaseColorId, _cachedBaseColor);
            SetMaterialFloat(material, OverlayAlphaId, _cachedOverlayAlpha);
            SetMaterialFloat(material, VisibilityGainId, _cachedVisibilityGain);
            SetMaterialFloat(material, MinVisibleBrightnessId, _cachedMinVisibleBrightness);
            return;
        }

        // Transparent blend (same appearance as OverlayTransparent) but ZWrite=1 so
        // the ATW compositor has per-pixel depth for correct reprojection on head movement.
        if (GetEffectiveRenderMode() == PointCloudRenderMode.OverlayDepthWrite)
        {
            material.renderQueue = _cachedRenderQueue;
            SetMaterialFloat(material, SrcBlendId,             _cachedSrcBlend);
            SetMaterialFloat(material, DstBlendId,             _cachedDstBlend);
            SetMaterialFloat(material, ZWriteId,               1.0f);   // depth writes ON
            SetMaterialFloat(material, ZTestId,                4.0f);   // LessEqual
            SetMaterialColor(material, BaseColorId,            _cachedBaseColor);
            SetMaterialFloat(material, OverlayAlphaId,         _cachedOverlayAlpha);
            SetMaterialFloat(material, VisibilityGainId,       _cachedVisibilityGain);
            SetMaterialFloat(material, MinVisibleBrightnessId, _cachedMinVisibleBrightness);
            return;
        }

        material.renderQueue = 2000;
        SetMaterialFloat(material, SrcBlendId, 1.0f);
        SetMaterialFloat(material, DstBlendId, 0.0f);
        SetMaterialFloat(material, ZWriteId, 1.0f);
        SetMaterialFloat(material, ZTestId, 4.0f);
        SetMaterialColor(material, BaseColorId, new Color(1.0f, 1.0f, 1.0f, 1.0f));
        SetMaterialFloat(material, OverlayAlphaId, 1.0f);
        SetMaterialFloat(material, VisibilityGainId, 1.0f);
        SetMaterialFloat(material, MinVisibleBrightnessId, 0.0f);
    }

    // Always use the renderMode field directly so Inspector and Bootstrap assignments
    // take effect. The previous _hasExplicitRenderModeSelection gate was never set
    // by anything, silently forcing OverlayTransparent (ZWrite=0) for every session.
    private PointCloudRenderMode GetEffectiveRenderMode() => renderMode;

    private void LogContractWarning(SourceState state, string message)
    {
        if (state == null)
            return;

        if (state.ContractWarningCount < 10)
            Debug.LogWarning($"[GpuMergedPointCloudLoader] {message}");
        state.ContractWarningCount++;
    }

    private void LogSourceFrameDiagnostics()
    {
        for (int i = 0; i < _sourceList.Count; i++)
        {
            var state = _sourceList[i];
            var frame = state.RenderFrame;
            if (state.LoggedFirstFrameDiagnostics || frame == null || frame.RawPoints == null || frame.ActualCount <= 0)
                continue;

            state.LoggedFirstFrameDiagnostics = true;
            Vector3 samplePoint = frame.RawPoints[0];
            Debug.Log(
                "[GpuMergedPointCloudLoader] First source frame "
                + $"name='{name}' source='{state.Label}' topic='{state.Topic}' actual={frame.ActualCount} declared={frame.DeclaredCapacity} "
                + $"samplePoint={samplePoint}"
            );
        }
    }

    private void BlockRuntime(string reason)
    {
        if (string.IsNullOrWhiteSpace(reason))
            reason = "Merged point-cloud renderer is unavailable.";

        if (string.IsNullOrWhiteSpace(_runtimeBlockReason))
            _runtimeBlockReason = reason;

        _runtimeReady = false;
    }

    private bool HasUploadResources(int requiredCount)
    {
        return instancedMaterial != null
            && instancedMaterial.shader != null
            && _matrixUpload != null
            && _colorUpload != null
            && _matrixBuffer != null
            && _colorBuffer != null
            && _matrixUpload.Length >= requiredCount
            && _colorUpload.Length >= requiredCount;
    }

    private bool HasDrawResources()
    {
        return instancedMaterial != null
            && instancedMaterial.shader != null
            && _cubeMesh != null
            && _matrixBuffer != null
            && _colorBuffer != null
            && _argsBuffer != null;
    }

    private void ClearCombinedDrawState()
    {
        _pointCount = 0;
        _activeSourceCount = 0;
    }

    private void LogRuntimeBlockedThrottled(string context)
    {
        float now = Time.unscaledTime;
        if (now < _nextRuntimeBlockedLogTime)
            return;

        _nextRuntimeBlockedLogTime = now + RuntimeBlockedLogIntervalSeconds;
        string reason = string.IsNullOrWhiteSpace(_runtimeBlockReason)
            ? "Merged point-cloud renderer is unavailable."
            : _runtimeBlockReason;
        Debug.LogError($"[GpuMergedPointCloudLoader] {context} for '{name}': {reason}");
    }

    private bool ShouldLogPostAlignDrawDiagnostics()
    {
        if (_sceneRoot == null || _sceneAnchor == null || _pointCount <= 0)
            return false;

        if (_sceneRoot.parent != null && !string.Equals(_sceneRoot.parent.name, "IRISNode", StringComparison.Ordinal))
            return true;

        if (_sceneRoot.localPosition.sqrMagnitude > 0.0001f)
            return true;

        if (Quaternion.Angle(_sceneRoot.localRotation, Quaternion.identity) > 0.1f)
            return true;

        return false;
    }

    private void LogPostAlignDrawDiagnostics(float effectiveCubeSize)
    {
        _loggedFirstPostAlignDrawDiagnostics = true;

        if (!TryComputeLocalPointStats(out Vector3 localMin, out Vector3 localMax, out Vector3 localCentroid))
        {
            Debug.LogWarning($"[GpuMergedPointCloudLoader] Unable to compute post-align point stats for '{name}'.");
            return;
        }

        Vector3 worldCentroid = _renderPoseMatrix.MultiplyPoint3x4(localCentroid);
        Debug.Log(
            "[GpuMergedPointCloudLoader] Post-align combined draw "
            + $"name='{name}' renderMode={GetEffectiveRenderMode()} points={_pointCount} activeSources={_activeSourceCount}/{_sourceList.Count} "
            + $"sceneRootPath='{GetTransformPath(_sceneRoot)}' sceneRootWorldPos={_sceneRoot.position} "
            + $"anchorPath='{GetTransformPath(_sceneAnchor)}' anchorWorldPos={_sceneAnchor.position} "
            + $"localCentroid={localCentroid} localMin={localMin} localMax={localMax} worldCentroid={worldCentroid} "
            + $"poseMatrix={FormatMatrix(_renderPoseMatrix)} cubeSize={effectiveCubeSize:F4} "
            + $"renderQueue={(instancedMaterial != null ? instancedMaterial.renderQueue : -1)} "
            + $"zWrite={FormatMaterialFlag(GetMaterialFloat(instancedMaterial, "_ZWrite", 0.0f))} "
            + $"zTest={FormatCompareFunction(GetMaterialFloat(instancedMaterial, "_ZTest", 4.0f))} "
            + $"blend={FormatBlendMode(GetMaterialFloat(instancedMaterial, "_SrcBlend", 1.0f))}/{FormatBlendMode(GetMaterialFloat(instancedMaterial, "_DstBlend", 0.0f))}"
        );

        if (!_loggedUploadedMatrixDiagnostics)
            LogUploadedMatrixDiagnostics();
    }

    private void LogUploadedMatrixDiagnostics()
    {
        _loggedUploadedMatrixDiagnostics = true;

        if (_matrixUpload == null || _pointCount <= 0)
        {
            Debug.LogWarning($"[GpuMergedPointCloudLoader] Uploaded matrix diagnostics unavailable for '{name}'.");
            return;
        }

        int sampleCount = Mathf.Min(3, Mathf.Min(_pointCount, _matrixUpload.Length));
        if (sampleCount <= 0)
        {
            Debug.LogWarning($"[GpuMergedPointCloudLoader] Uploaded matrix diagnostics found no samples for '{name}'.");
            return;
        }

        var sampleBuilder = new StringBuilder(256);
        for (int i = 0; i < sampleCount; i++)
        {
            if (i > 0)
                sampleBuilder.Append(' ');

            Vector3 localTranslation = ExtractTranslation(_matrixUpload[i]);
            Vector3 worldTranslation = _renderPoseMatrix.MultiplyPoint3x4(localTranslation);
            sampleBuilder.Append($"#{i}:local={localTranslation} world={worldTranslation}");
        }

        Debug.Log(
            "[GpuMergedPointCloudLoader] Uploaded matrix samples "
            + $"name='{name}' points={_pointCount} samples={sampleBuilder}"
        );
    }

    private bool TryComputeLocalPointStats(out Vector3 localMin, out Vector3 localMax, out Vector3 localCentroid)
    {
        localMin = Vector3.zero;
        localMax = Vector3.zero;
        localCentroid = Vector3.zero;

        if (_matrixUpload == null || _pointCount <= 0)
            return false;

        int safeCount = Mathf.Min(_pointCount, _matrixUpload.Length);
        if (safeCount <= 0)
            return false;

        Vector3 firstPoint = ExtractTranslation(_matrixUpload[0]);
        localMin = firstPoint;
        localMax = firstPoint;
        Vector3 sum = Vector3.zero;
        for (int i = 0; i < safeCount; i++)
        {
            Vector3 localPoint = ExtractTranslation(_matrixUpload[i]);
            localMin = Vector3.Min(localMin, localPoint);
            localMax = Vector3.Max(localMax, localPoint);
            sum += localPoint;
        }

        localCentroid = sum / safeCount;
        return true;
    }

    private string FormatTopicList()
    {
        if (_sourceList.Count <= 0)
            return "[]";

        var builder = new StringBuilder(64);
        builder.Append('[');
        for (int i = 0; i < _sourceList.Count; i++)
        {
            if (i > 0)
                builder.Append(", ");
            builder.Append(_sourceList[i].Topic);
        }
        builder.Append(']');
        return builder.ToString();
    }

    private static string BuildSourceLabel(string topic)
    {
        if (string.IsNullOrWhiteSpace(topic))
            return "unknown";

        string[] parts = topic.Split('/');
        if (parts.Length >= 3 && !string.IsNullOrWhiteSpace(parts[2]))
            return parts[2];
        return topic;
    }

    private static float GetMaterialFloat(Material material, int propertyId, float fallbackValue)
    {
        if (material == null || !material.HasProperty(propertyId))
            return fallbackValue;

        return material.GetFloat(propertyId);
    }

    private static float GetMaterialFloat(Material material, string propertyName, float fallbackValue)
    {
        if (material == null || string.IsNullOrWhiteSpace(propertyName) || !material.HasProperty(propertyName))
            return fallbackValue;

        return material.GetFloat(propertyName);
    }

    private static Color GetMaterialColor(Material material, int propertyId, Color fallbackValue)
    {
        if (material == null || !material.HasProperty(propertyId))
            return fallbackValue;

        return material.GetColor(propertyId);
    }

    private static void SetMaterialFloat(Material material, int propertyId, float value)
    {
        if (material == null || !material.HasProperty(propertyId))
            return;

        material.SetFloat(propertyId, value);
    }

    private static void SetMaterialColor(Material material, int propertyId, Color value)
    {
        if (material == null || !material.HasProperty(propertyId))
            return;

        material.SetColor(propertyId, value);
    }

    private static string FormatMaterialFlag(float value)
    {
        return Mathf.RoundToInt(value) != 0 ? "On" : "Off";
    }

    private static string FormatCompareFunction(float value)
    {
        switch (Mathf.RoundToInt(value))
        {
            case 0: return "Disabled";
            case 1: return "Never";
            case 2: return "Less";
            case 3: return "Equal";
            case 4: return "LessEqual";
            case 5: return "Greater";
            case 6: return "NotEqual";
            case 7: return "GreaterEqual";
            case 8: return "Always";
            default: return $"Unknown({value:F0})";
        }
    }

    private static string FormatBlendMode(float value)
    {
        switch (Mathf.RoundToInt(value))
        {
            case 0: return "Zero";
            case 1: return "One";
            case 2: return "DstColor";
            case 3: return "SrcColor";
            case 4: return "OneMinusDstColor";
            case 5: return "SrcAlpha";
            case 6: return "OneMinusSrcColor";
            case 7: return "DstAlpha";
            case 8: return "OneMinusDstAlpha";
            case 9: return "SrcAlphaSaturate";
            case 10: return "OneMinusSrcAlpha";
            default: return $"Unknown({value:F0})";
        }
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

    private static Vector3 ExtractTranslation(Matrix4x4 matrix)
    {
        return new Vector3(matrix.m03, matrix.m13, matrix.m23);
    }
}
