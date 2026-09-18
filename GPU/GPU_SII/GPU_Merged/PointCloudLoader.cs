using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using IRIS.SceneLoader;
using NetMQ;
using NetMQ.Sockets;
using UnityEngine;
using UnityEngine.Rendering;

public enum PointCloudRenderMode
{
    DepthTestedOpaque = 0,
    OverlayTransparent = 1,
}

[DisallowMultipleComponent]
[RequireComponent(typeof(MeshFilter), typeof(MeshRenderer))]
public class PointCloudTest : MonoBehaviour
{
    private const string LoaderBuildStamp = "2026-03-17-gpu-capacity-100k";

    [Header("Connection")]
    public string publisherIp = "192.168.0.208";
    public int topicPort = 7741;
    public string topic = "SimPub/Sensors/top/pc";
    public int maxPoints = 100000;
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
    public float minReasonableViewerY = -2.0f;
    public float maxReasonableViewerY = 3.5f;

    [Header("Diagnostic Probes")]
    public bool enableDiagnosticProbeMode = false;
    public int diagnosticProbePointCount = 24;
    public float diagnosticProbeCubeSize = 0.02f;

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
    private const float RuntimeBlockedLogIntervalSeconds = 5.0f;

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

            RawPoints = new Vector3[capacity];
            Colors = new Color32[capacity];
            Capacity = capacity;
        }
    }

    private readonly object _frameLock = new object();
    private readonly object _netMqLock = new object();
    private readonly NetMQMessage _netMessage = new NetMQMessage();

    private SubscriberSocket _sub;
    private NetMQPoller _poller;
    private volatile bool _shuttingDown;
    private long _lastPayloadTick;
    private float _nextReconnectTime;

    private DecodedFrame _decodeFrame = new DecodedFrame();
    private DecodedFrame _pendingFrame;
    private DecodedFrame _renderFrame;

    private Mesh _cubeMesh;
    private MeshRenderer _meshRenderer;
    private Material _runtimeMaterial;
    private ComputeBuffer _matrixBuffer;
    private ComputeBuffer _colorBuffer;
    private ComputeBuffer _argsBuffer;
    private readonly uint[] _argsData = new uint[5];
    private Matrix4x4[] _matrixUpload;
    private Vector4[] _colorUpload;
    private Bounds _renderBounds;
    private Matrix4x4 _renderPoseMatrix = Matrix4x4.identity;
    private int _allocatedCapacity;
    private int _pointCount;
    private int _bufferAllocationCount;
    private int _contractWarningCount;
    private bool _runtimeReady = true;
    private string _runtimeBlockReason;
    private bool _loggedRuntimeCapabilities;
    private bool _loggedFirstFrameDiagnostics;
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
    private readonly List<GameObject> _diagnosticProbeObjects = new List<GameObject>();
    private bool _loggedDiagnosticProbeIndices;
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
        EnsureRendererResources();
        TryAttachToSceneRoot();
        UpdateRenderPose();
        _lastPayloadTick = System.Diagnostics.Stopwatch.GetTimestamp();
    }

    private void Start()
    {
        StartNetMq();
    }

    private void Update()
    {
        TryAttachToSceneRoot();
        UpdateRenderPose();
        ApplyPendingFrame();
        TryReconnectIfIdle();
        UpdateDiagnosticProbes();
        ApplyRuntimeMaterialMode();
        DrawPointCloud();
    }

    private void OnApplicationQuit()
    {
        _shuttingDown = true;
        ShutdownNetMq();
    }

    private void OnDestroy()
    {
        _shuttingDown = true;
        ShutdownNetMq();
        ReleaseBuffers();
        DestroyDiagnosticProbes();
        DestroyRuntimeMaterial();
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
            string summary =
                "[PointCloudTest] "
                + $"build='{LoaderBuildStamp}' name='{name}' api={SystemInfo.graphicsDeviceType} "
                + $"maxPoints={maxPoints} "
                + $"renderMode={GetEffectiveRenderMode()} "
                + $"supportsInstancing={SystemInfo.supportsInstancing} "
                + $"material={(instancedMaterial != null ? instancedMaterial.name : "null")} "
                + $"shader={(instancedMaterial != null && instancedMaterial.shader != null ? instancedMaterial.shader.name : "null")}";

            if (_runtimeReady)
                Debug.Log(summary);
            else
                Debug.LogError($"{summary} blocked='{_runtimeBlockReason}'");
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
                Debug.LogError($"[PointCloudTest] {_runtimeBlockReason}");
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
                Debug.LogError($"[PointCloudTest] {_runtimeBlockReason}");
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
            Debug.LogError($"[PointCloudTest] {_runtimeBlockReason}");
            return;
        }

        _runtimeMaterial = new Material(shader)
        {
            name = "CubeRenderingMaterial"
        };
        _runtimeMaterial.hideFlags = HideFlags.DontSave;
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
        instancedMaterial.SetMatrix(RenderPoseMatrixId, _renderPoseMatrix);
        return true;
    }

    private void EvaluateRuntimeSupport()
    {
        if (!_runtimeReady)
            return;

        if (instancedMaterial == null)
        {
            BlockRuntime("No instanced material is available for GPU_Merged point-cloud rendering.");
            return;
        }

        if (!SystemInfo.supportsInstancing)
        {
            BlockRuntime("GPU_Merged point-cloud rendering requires instancing support.");
        }
    }

    private void EnsureCapacity(int declaredCapacity)
    {
        if (!_runtimeReady)
            return;

        declaredCapacity = Mathf.Max(1, declaredCapacity);
        if (_matrixBuffer != null
            && _colorBuffer != null
            && _argsBuffer != null
            && _allocatedCapacity >= declaredCapacity)
        {
            return;
        }

        ReleaseBuffers();
        _allocatedCapacity = declaredCapacity;
        _matrixUpload = new Matrix4x4[_allocatedCapacity];
        _colorUpload = new Vector4[_allocatedCapacity];
        _matrixBuffer = new ComputeBuffer(_allocatedCapacity, sizeof(float) * 16);
        _colorBuffer = new ComputeBuffer(_allocatedCapacity, sizeof(float) * 4);
        _argsBuffer = new ComputeBuffer(1, _argsData.Length * sizeof(uint), ComputeBufferType.IndirectArguments);
        _bufferAllocationCount++;

        if (logBufferLifecycle)
        {
            Debug.Log(
                $"[PointCloudTest] Allocated buffers for '{name}' capacity={_allocatedCapacity} allocationCount={_bufferAllocationCount}."
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
        if (!e.Socket.TryReceiveMultipartMessage(ref msg) || msg.FrameCount < 2)
            return;

        string receivedTopic;
        try { receivedTopic = msg[0].ConvertToString(); }
        catch { return; }

        if (!string.Equals(receivedTopic, topic, StringComparison.Ordinal))
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
                $"Dropped '{receivedTopic}' because declared capacity {header.DeclaredCapacity} exceeds maxPoints {maxPoints}."
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
                false,
                fallbackBaseColor,
                out error))
        {
            LogContractWarning(error);
            return;
        }

        frame.DeclaredCapacity = header.DeclaredCapacity;
        frame.ActualCount = header.ActualCount;

        Interlocked.Exchange(ref _lastPayloadTick, System.Diagnostics.Stopwatch.GetTimestamp());
        lock (_frameLock)
        {
            var previousPending = _pendingFrame;
            _pendingFrame = frame;
            _decodeFrame = previousPending != null && !ReferenceEquals(previousPending, frame)
                ? previousPending
                : new DecodedFrame();
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

        if (frame.ActualCount <= 0)
        {
            Clear();
            return;
        }

        if (!_runtimeReady)
        {
            Clear();
            LogRuntimeBlockedThrottled("Skipping point-cloud frame upload");
            return;
        }

        EnsureCapacity(frame.DeclaredCapacity);
        if (!HasUploadResources(frame.ActualCount))
        {
            BlockRuntime("GPU point-cloud upload resources are unavailable.");
            Clear();
            LogRuntimeBlockedThrottled("Skipping point-cloud frame upload");
            return;
        }

        float safeCubeSize = Mathf.Max(0.0001f, ResolveEffectiveCubeSize() * scale);
        for (int i = 0; i < frame.ActualCount; i++)
        {
            Vector3 point = frame.RawPoints[i] + debugOffset;
            _matrixUpload[i] = Matrix4x4.TRS(point * scale, Quaternion.identity, Vector3.one * safeCubeSize);

            Color32 color = frame.Colors[i];
            _colorUpload[i] = new Vector4(
                color.r / 255.0f,
                color.g / 255.0f,
                color.b / 255.0f,
                1.0f
            );
        }

        _matrixBuffer.SetData(_matrixUpload, 0, 0, frame.ActualCount);
        _colorBuffer.SetData(_colorUpload, 0, 0, frame.ActualCount);
        if (!BindMaterialResources())
        {
            BlockRuntime("GPU point-cloud material bindings are unavailable.");
            Clear();
            LogRuntimeBlockedThrottled("Skipping point-cloud frame upload");
            return;
        }
        _pointCount = frame.ActualCount;

        if (!_loggedFirstFrameDiagnostics)
        {
            _loggedFirstFrameDiagnostics = true;
            Vector3 samplePoint = frame.RawPoints[0];
            string anchorPath = _sceneAnchor != null ? GetTransformPath(_sceneAnchor) : "<none>";
            Debug.Log(
                "[PointCloudTest] First frame "
                + $"build='{LoaderBuildStamp}' name='{name}' topic='{topic}' actual={frame.ActualCount} declared={frame.DeclaredCapacity} "
                + $"samplePoint={samplePoint} anchorPath='{anchorPath}' bufferCapacity={_allocatedCapacity} "
                + $"allocationCount={_bufferAllocationCount}"
            );
        }
    }

    private void DrawPointCloud()
    {
        if (!_runtimeReady)
        {
            if (_pointCount > 0)
                LogRuntimeBlockedThrottled("Skipping point-cloud draw");
            return;
        }

        if (!renderEnabled)
            return;

        if (_pointCount <= 0)
            return;

        if (!IsReadyForStableAnchoredDraw())
            return;

        if (!HasDrawResources())
        {
            BlockRuntime("GPU point-cloud draw resources are unavailable.");
            Clear();
            LogRuntimeBlockedThrottled("Skipping point-cloud draw");
            return;
        }

        float effectiveCubeSize = Mathf.Max(0.0001f, ResolveEffectiveCubeSize() * scale);
        _renderBounds.center = ExtractTranslation(_renderPoseMatrix);
        _renderBounds.size = Vector3.one * Mathf.Max(1.0f, renderBoundsSize);
        if (!BindMaterialResources())
        {
            BlockRuntime("GPU point-cloud draw material bindings are unavailable.");
            Clear();
            LogRuntimeBlockedThrottled("Skipping point-cloud draw");
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
            null,
            ShadowCastingMode.Off,
            false,
            gameObject.layer
        );

        if (!_loggedFirstDrawDiagnostics)
        {
            _loggedFirstDrawDiagnostics = true;
            Debug.Log(
                "[PointCloudTest] First draw "
                + $"name='{name}' renderMode={GetEffectiveRenderMode()} points={_pointCount} boundsCenter={_renderBounds.center} "
                + $"boundsSize={_renderBounds.size} poseMatrix={FormatMatrix(_renderPoseMatrix)}"
            );
        }

        if (_sceneAnchor != null && !_loggedFirstAnchoredDrawDiagnostics)
        {
            _loggedFirstAnchoredDrawDiagnostics = true;
            Debug.Log(
                "[PointCloudTest] First anchored draw "
                + $"name='{name}' renderMode={GetEffectiveRenderMode()} anchorPath='{GetTransformPath(_sceneAnchor)}' points={_pointCount} "
                + $"boundsCenter={_renderBounds.center} poseMatrix={FormatMatrix(_renderPoseMatrix)}"
            );
        }

        if (!_loggedMaterialStateDiagnostics)
        {
            _loggedMaterialStateDiagnostics = true;
            Debug.Log(
                "[PointCloudTest] Material state "
                + $"name='{name}' renderMode={GetEffectiveRenderMode()} material='{(instancedMaterial != null ? instancedMaterial.name : "<none>")}' "
                + $"shader='{(instancedMaterial != null && instancedMaterial.shader != null ? instancedMaterial.shader.name : "<none>")}' "
                + $"renderQueue={(instancedMaterial != null ? instancedMaterial.renderQueue : -1)} "
                + $"zWrite={FormatMaterialFlag(GetMaterialFloat(instancedMaterial, "_ZWrite", 0.0f))} "
                + $"zTest={FormatCompareFunction(GetMaterialFloat(instancedMaterial, "_ZTest", 4.0f))} "
                + $"blend={FormatBlendMode(GetMaterialFloat(instancedMaterial, "_SrcBlend", 1.0f))}/{FormatBlendMode(GetMaterialFloat(instancedMaterial, "_DstBlend", 0.0f))} "
                + $"cubeSize={effectiveCubeSize:F4} pointCount={_pointCount}"
            );
        }

        if (!_loggedFirstPostAlignDrawDiagnostics && ShouldLogPostAlignDrawDiagnostics())
            LogPostAlignDrawDiagnostics(effectiveCubeSize);
    }

    public void Clear()
    {
        _pointCount = 0;
        _renderFrame = null;
        SetDiagnosticProbesActive(false);
    }

    public void ResetBootstrapBindingState()
    {
        _sceneRoot = null;
        _sceneAnchor = null;
        _sceneAnchorName = string.Empty;
        _loggedAttachDiagnostics = false;
        _loggedWaitingForSceneAnchor = false;
        _loggedFirstAnchoredDrawDiagnostics = false;
        _loggedFirstPostAlignDrawDiagnostics = false;
        _loggedMaterialStateDiagnostics = false;
        _loggedUploadedMatrixDiagnostics = false;
        _loggedDrawStabilityWaiting = false;
        _loggedDiagnosticProbeIndices = false;
        _firstDrawUnlocked = false;
        _stableDrawChecks = 0;
        SetDiagnosticProbesActive(false);
    }

    public void SetRenderMode(PointCloudRenderMode value)
    {
        renderMode = value;
        _hasExplicitRenderModeSelection = true;
        ApplyRuntimeMaterialMode();
    }

    private void TryReconnectIfIdle()
    {
        if (!autoReconnectOnIdle || Time.unscaledTime < _nextReconnectTime)
            return;

        double idleFor =
            (System.Diagnostics.Stopwatch.GetTimestamp() - Interlocked.Read(ref _lastPayloadTick))
            * TimestampToSeconds;
        if (idleFor < Mathf.Max(0.5f, reconnectIdleSeconds))
            return;

        if (_pointCount > 0 || _renderFrame != null)
        {
            Debug.LogWarning(
                $"[PointCloudTest] Clearing stale rendered cloud on '{name}' after {idleFor:F2}s idle for '{topic}'."
            );
            Clear();
        }

        Debug.LogWarning($"[PointCloudTest] Idle for {idleFor:F2}s on '{name}'. Reconnecting '{topic}'.");
        ReconnectNetMq();
    }

    private void StartNetMq()
    {
        var sub = new SubscriberSocket();
        sub.Options.ReceiveHighWatermark = 2;
        sub.Options.Linger = TimeSpan.Zero;
        sub.Connect($"tcp://{publisherIp}:{topicPort}");
        sub.Subscribe(topic);
        sub.ReceiveReady += OnMsg;

        var poller = new NetMQPoller { sub };
        lock (_netMqLock)
        {
            _sub = sub;
            _poller = poller;
        }

        poller.RunAsync();
        Interlocked.Exchange(ref _lastPayloadTick, System.Diagnostics.Stopwatch.GetTimestamp());
        _nextReconnectTime = Time.unscaledTime + Mathf.Max(0.5f, reconnectCooldownSeconds);
        Debug.Log($"[PointCloudTest] Connected tcp://{publisherIp}:{topicPort} topic='{topic}' for '{name}'.");
    }

    private void ReconnectNetMq()
    {
        ShutdownNetMq();
        _shuttingDown = false;
        StartNetMq();
    }

    private void ShutdownNetMq()
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
        try { if (poller != null && poller.IsRunning) poller.Stop(); } catch { }
        try { sub?.Close(); } catch { }
        try { sub?.Dispose(); } catch { }
        try { poller?.Dispose(); } catch { }
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
                Debug.LogWarning($"[PointCloudTest] Waiting for SimScene '{simSceneName}' before attaching '{name}'.");
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
                $"[PointCloudTest] Bound '{name}' to SimScene '{root.name}' via anchor '{anchor.name}' path='{GetTransformPath(anchor)}'."
            );
        }
    }

    private void UpdateRenderPose()
    {
        if (parentToSimScene && _sceneAnchor != null)
            _renderPoseMatrix = _sceneAnchor.localToWorldMatrix;
        else
            _renderPoseMatrix = transform.localToWorldMatrix;

        _renderBounds.center = ExtractTranslation(_renderPoseMatrix);
        _renderBounds.size = Vector3.one * Mathf.Max(1.0f, renderBoundsSize);
        if (instancedMaterial != null && instancedMaterial.shader != null)
            instancedMaterial.SetMatrix(RenderPoseMatrixId, _renderPoseMatrix);
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

        string fragment = name;
        if (!string.IsNullOrWhiteSpace(topic))
        {
            string[] parts = topic.Split('/');
            if (parts.Length >= 3 && !string.IsNullOrWhiteSpace(parts[2]))
                fragment = parts[2];
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
                $"[PointCloudTest] XR recovery detected for '{name}'. Waiting for stable anchored draw before resuming point-cloud rendering."
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
                "[PointCloudTest] Draw gate released "
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
                "[PointCloudTest] Waiting for stable anchored draw "
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

    private void UpdateDiagnosticProbes()
    {
        if (!enableDiagnosticProbeMode
            || _sceneAnchor == null
            || _renderFrame == null
            || _renderFrame.ActualCount <= 0)
        {
            SetDiagnosticProbesActive(false);
            return;
        }

        int probeCount = Mathf.Clamp(
            diagnosticProbePointCount,
            0,
            Mathf.Min(_renderFrame.ActualCount, _renderFrame.RawPoints != null ? _renderFrame.RawPoints.Length : 0)
        );
        if (probeCount <= 0)
        {
            SetDiagnosticProbesActive(false);
            return;
        }

        EnsureDiagnosticProbes(probeCount);
        float probeScale = Mathf.Max(0.001f, diagnosticProbeCubeSize);
        if (!_loggedDiagnosticProbeIndices)
            LogDiagnosticProbeIndices(probeCount);
        for (int i = 0; i < _diagnosticProbeObjects.Count; i++)
        {
            var probe = _diagnosticProbeObjects[i];
            if (probe == null)
                continue;

            bool shouldShow = i < probeCount;
            probe.SetActive(shouldShow);
            if (!shouldShow)
                continue;

            var probeTransform = probe.transform;
            if (probeTransform.parent != _sceneAnchor)
                probeTransform.SetParent(_sceneAnchor, false);

            int sampleIndex = ResolveDiagnosticProbeIndex(i, probeCount);
            probeTransform.localPosition = (_renderFrame.RawPoints[sampleIndex] + debugOffset) * scale;
            probeTransform.localRotation = Quaternion.identity;
            probeTransform.localScale = Vector3.one * probeScale;
        }
    }

    private int ResolveDiagnosticProbeIndex(int probeIndex, int probeCount)
    {
        if (_renderFrame == null || _renderFrame.RawPoints == null || _renderFrame.ActualCount <= 0)
            return 0;

        int safeCount = Mathf.Min(_renderFrame.ActualCount, _renderFrame.RawPoints.Length);
        if (safeCount <= 1 || probeCount <= 1)
            return 0;

        int clampedProbeIndex = Mathf.Clamp(probeIndex, 0, probeCount - 1);
        return Mathf.Clamp((clampedProbeIndex * safeCount) / probeCount, 0, safeCount - 1);
    }

    private void LogDiagnosticProbeIndices(int probeCount)
    {
        if (_renderFrame == null || _renderFrame.RawPoints == null || _renderFrame.ActualCount <= 0)
            return;

        _loggedDiagnosticProbeIndices = true;
        int safeCount = Mathf.Min(_renderFrame.ActualCount, _renderFrame.RawPoints.Length);
        var builder = new StringBuilder(128);
        builder.Append('[');
        for (int i = 0; i < probeCount; i++)
        {
            if (i > 0)
                builder.Append(", ");
            builder.Append(ResolveDiagnosticProbeIndex(i, probeCount));
        }
        builder.Append(']');
        Debug.Log(
            $"[PointCloudTest] Diagnostic probe indices name='{name}' probes={probeCount} actual={safeCount} indices={builder}"
        );
    }

    private void EnsureDiagnosticProbes(int count)
    {
        while (_diagnosticProbeObjects.Count < count)
        {
            var probe = GameObject.CreatePrimitive(PrimitiveType.Cube);
            probe.name = $"{name}_DiagnosticProbe_{_diagnosticProbeObjects.Count}";
            probe.hideFlags = HideFlags.DontSave;
            var collider = probe.GetComponent<Collider>();
            if (collider != null)
            {
                if (Application.isPlaying)
                    Destroy(collider);
                else
                    DestroyImmediate(collider);
            }

            var probeRenderer = probe.GetComponent<Renderer>();
            if (probeRenderer != null)
            {
                var shader = Shader.Find("Unlit/Color");
                if (shader == null)
                    shader = Shader.Find("Standard");
                if (shader != null)
                {
                    var probeMaterial = new Material(shader)
                    {
                        name = $"{probe.name}_Material"
                    };
                    if (probeMaterial.HasProperty("_Color"))
                        probeMaterial.color = Color.yellow;
                    probeRenderer.sharedMaterial = probeMaterial;
                }
            }

            _diagnosticProbeObjects.Add(probe);
        }
    }

    private void SetDiagnosticProbesActive(bool active)
    {
        for (int i = 0; i < _diagnosticProbeObjects.Count; i++)
        {
            var probe = _diagnosticProbeObjects[i];
            if (probe != null)
                probe.SetActive(active && enableDiagnosticProbeMode);
        }
    }

    private void DestroyDiagnosticProbes()
    {
        for (int i = 0; i < _diagnosticProbeObjects.Count; i++)
        {
            var probe = _diagnosticProbeObjects[i];
            if (probe == null)
                continue;

            var probeRenderer = probe.GetComponent<Renderer>();
            if (probeRenderer != null && probeRenderer.sharedMaterial != null)
            {
                if (Application.isPlaying)
                    Destroy(probeRenderer.sharedMaterial);
                else
                    DestroyImmediate(probeRenderer.sharedMaterial);
            }

            if (Application.isPlaying)
                Destroy(probe);
            else
                DestroyImmediate(probe);
        }
        _diagnosticProbeObjects.Clear();
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

    private PointCloudRenderMode GetEffectiveRenderMode()
    {
        if (_hasExplicitRenderModeSelection)
            return renderMode;

        // Older scene data only serialized the debug toggle, so preserve that
        // behavior until the new enum-backed mode is explicitly stored.
        return useDepthTestDebugMode
            ? PointCloudRenderMode.DepthTestedOpaque
            : PointCloudRenderMode.OverlayTransparent;
    }

    private void LogContractWarning(string message)
    {
        if (_contractWarningCount < 10)
            Debug.LogWarning($"[PointCloudTest] {message}");
        _contractWarningCount++;
    }

    private void BlockRuntime(string reason)
    {
        if (string.IsNullOrWhiteSpace(reason))
            reason = "Point-cloud renderer is unavailable.";

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

    private void LogRuntimeBlockedThrottled(string context)
    {
        float now = Time.unscaledTime;
        if (now < _nextRuntimeBlockedLogTime)
            return;

        _nextRuntimeBlockedLogTime = now + RuntimeBlockedLogIntervalSeconds;
        string reason = string.IsNullOrWhiteSpace(_runtimeBlockReason)
            ? "Point-cloud renderer is unavailable."
            : _runtimeBlockReason;
        Debug.LogError($"[PointCloudTest] {context} for '{name}': {reason}");
    }

    private bool ShouldLogPostAlignDrawDiagnostics()
    {
        if (_sceneRoot == null || _sceneAnchor == null || _renderFrame == null || _pointCount <= 0)
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
            Debug.LogWarning($"[PointCloudTest] Unable to compute post-align point stats for '{name}'.");
            return;
        }

        Vector3 worldCentroid = _renderPoseMatrix.MultiplyPoint3x4(localCentroid);
        Debug.Log(
            "[PointCloudTest] Post-align draw "
            + $"name='{name}' renderMode={GetEffectiveRenderMode()} points={_pointCount} "
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
            Debug.LogWarning($"[PointCloudTest] Uploaded matrix diagnostics unavailable for '{name}'.");
            return;
        }

        int sampleCount = Mathf.Min(3, Mathf.Min(_pointCount, _matrixUpload.Length));
        if (sampleCount <= 0)
        {
            Debug.LogWarning($"[PointCloudTest] Uploaded matrix diagnostics found no samples for '{name}'.");
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
            "[PointCloudTest] Uploaded matrix samples "
            + $"name='{name}' points={_pointCount} samples={sampleBuilder}"
        );
    }

    private bool TryComputeLocalPointStats(out Vector3 localMin, out Vector3 localMax, out Vector3 localCentroid)
    {
        localMin = Vector3.zero;
        localMax = Vector3.zero;
        localCentroid = Vector3.zero;

        if (_renderFrame == null || _renderFrame.RawPoints == null || _pointCount <= 0)
            return false;

        int safeCount = Mathf.Min(_pointCount, _renderFrame.RawPoints.Length);
        if (safeCount <= 0)
            return false;

        Vector3 firstPoint = (_renderFrame.RawPoints[0] + debugOffset) * scale;
        localMin = firstPoint;
        localMax = firstPoint;
        Vector3 sum = Vector3.zero;
        for (int i = 0; i < safeCount; i++)
        {
            Vector3 localPoint = (_renderFrame.RawPoints[i] + debugOffset) * scale;
            localMin = Vector3.Min(localMin, localPoint);
            localMax = Vector3.Max(localMax, localPoint);
            sum += localPoint;
        }

        localCentroid = sum / safeCount;
        return true;
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
