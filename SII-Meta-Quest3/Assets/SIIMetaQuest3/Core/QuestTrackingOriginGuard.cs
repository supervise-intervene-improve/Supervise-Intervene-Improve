using System;
using UnityEngine;

[DefaultExecutionOrder(9000)]
public class QuestTrackingOriginGuard : MonoBehaviour
{
    public static QuestTrackingOriginGuard Instance { get; private set; }

    [Header("Runtime Scope")]
    public bool runInEditor = false;
    public bool androidOnly = true;

    [Header("Desired XR State")]
    public OVRManager.TrackingOrigin desiredTrackingOrigin = OVRManager.TrackingOrigin.Stage;
    public bool desiredAllowRecenter = false;

    [Header("Stability")]
    public float settleWindowSeconds = 1.5f;
    public bool monitorHeadPoseJumps = false;
    public float headPoseJumpMeters = 1.0f;
    public string preferredHeadTransformName = "CenterEyeAnchor";

    [Header("World Frame")]
    public string preferredRigRootName = "[BuildingBlock] Camera Rig";
    public string preferredTrackingSpaceName = "TrackingSpace";
    public float maxRigRootWorldDriftMeters = 0.25f;
    public float maxTrackingSpaceWorldDriftMeters = 0.25f;
    public bool snapRigRootToBaselineWhenDriftPersists = false;
    public float snapRigRootDriftMeters = 1.0f;
    public bool disableLocomotorOnStart = true;
    public string locomotorObjectName = "Locomotor";
    public float worldFrameLogIntervalSeconds = 2.0f;

    [Header("Monitoring")]
    public float checkIntervalSeconds = 0.25f;
    public float statusLogIntervalSeconds = 5.0f;
    public bool logStableStatus = true;

    private float _nextCheckAt;
    private float _nextStatusLogAt;
    private int _reapplyCount;
    private float _stableAfterTime;
    private Transform _headTransform;
    private Transform _rigRootTransform;
    private Transform _trackingSpaceTransform;
    private bool _hasLastHeadWorldPosition;
    private Vector3 _lastHeadWorldPosition;
    private bool _eventsSubscribed;
    private bool _runtimeLossActive;
    private bool _runtimeRecoveryPending;
    private float _lastLossSignalAt = float.NegativeInfinity;
    private float _lastRecoverySignalAt = float.NegativeInfinity;
    private string _lastLossReason = string.Empty;
    private string _lastRecoveryReason = string.Empty;
    private bool _hasWorldFrameBaseline;
    private bool _worldFrameDriftActive;
    private bool _loggedLocomotorStatus;
    private Vector3 _baselineRigRootWorldPosition;
    private Quaternion _baselineRigRootWorldRotation = Quaternion.identity;
    private Vector3 _baselineTrackingSpaceWorldPosition;
    private Vector3 _currentRigRootWorldDelta = Vector3.zero;
    private Vector3 _currentTrackingSpaceWorldDelta = Vector3.zero;
    private float _nextWorldFrameLogAt = float.NegativeInfinity;
    private const float EventDedupeSeconds = 0.5f;

    public bool IsRuntimeStable
    {
        get
        {
            if (!ShouldRunInCurrentRuntime())
                return true;
            if (_runtimeLossActive)
                return false;
            if (_runtimeRecoveryPending && Time.unscaledTime < _stableAfterTime)
                return false;
            return true;
        }
    }
    public bool IsWorldFrameStable
    {
        get
        {
            if (!ShouldRunInCurrentRuntime())
                return true;
            if (!IsRuntimeStable)
                return false;
            return _hasWorldFrameBaseline && !_worldFrameDriftActive;
        }
    }

    public bool HasWorldFrameBaseline => _hasWorldFrameBaseline;
    public Vector3 CurrentRigRootWorldDelta => _currentRigRootWorldDelta;
    public Vector3 CurrentTrackingSpaceWorldDelta => _currentTrackingSpaceWorldDelta;
    public Transform RigRootTransform => ResolveRigRootTransform();
    public Transform TrackingSpaceTransform => ResolveTrackingSpaceTransform();
    public float LastUnstableAt { get; private set; } = float.NegativeInfinity;
    public float LastRecoveryAt { get; private set; } = float.NegativeInfinity;
    public float LastWorldFrameBaselineAt { get; private set; } = float.NegativeInfinity;
    public float LastWorldFrameDriftAt { get; private set; } = float.NegativeInfinity;

    public static QuestTrackingOriginGuard ResolveSharedGuard()
    {
        if (Instance != null)
            return Instance;
        return FindAnyObjectByType<QuestTrackingOriginGuard>();
    }

    private void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Debug.LogWarning(
                $"[QuestTrackingOriginGuard] Multiple instances detected. Keeping the first instance on '{Instance.gameObject.name}'."
            );
            enabled = false;
            return;
        }

        Instance = this;
    }

    private void OnEnable()
    {
        if (Instance != null && Instance != this)
            return;

        if (Instance == null)
            Instance = this;

        if (!ShouldRunInCurrentRuntime())
            return;

        OVRManager.HMDAcquired += OnHMDAcquired;
        OVRManager.VrFocusAcquired += OnVrFocusAcquired;
        OVRManager.InputFocusAcquired += OnInputFocusAcquired;
        OVRManager.TrackingAcquired += OnTrackingAcquired;

        OVRManager.HMDLost += OnHMDLost;
        OVRManager.VrFocusLost += OnVrFocusLost;
        OVRManager.InputFocusLost += OnInputFocusLost;
        OVRManager.TrackingLost += OnTrackingLost;
        _eventsSubscribed = true;
    }

    private void OnDisable()
    {
        if (!_eventsSubscribed)
            return;

        OVRManager.HMDAcquired -= OnHMDAcquired;
        OVRManager.VrFocusAcquired -= OnVrFocusAcquired;
        OVRManager.InputFocusAcquired -= OnInputFocusAcquired;
        OVRManager.TrackingAcquired -= OnTrackingAcquired;

        OVRManager.HMDLost -= OnHMDLost;
        OVRManager.VrFocusLost -= OnVrFocusLost;
        OVRManager.InputFocusLost -= OnInputFocusLost;
        OVRManager.TrackingLost -= OnTrackingLost;
        _eventsSubscribed = false;
    }

    private void OnDestroy()
    {
        if (Instance == this)
            Instance = null;
    }

    private void Start()
    {
        if (!ShouldRunInCurrentRuntime())
        {
            enabled = false;
            return;
        }

        EnsureLocomotorDisabled();
        EnforceSettings("Start", logWhenAlreadyCorrect: true);
        UpdateWorldFrameState("Start");
    }

    private void Update()
    {
        if (!ShouldRunInCurrentRuntime())
            return;

        if (_runtimeRecoveryPending && Time.unscaledTime >= _stableAfterTime)
            _runtimeRecoveryPending = false;

        if (Time.unscaledTime >= _nextCheckAt)
        {
            _nextCheckAt = Time.unscaledTime + Mathf.Max(0.05f, checkIntervalSeconds);
            EnsureLocomotorDisabled();
            EnforceSettings("Watchdog");
            UpdateWorldFrameState("Watchdog");
        }

        if (monitorHeadPoseJumps)
            UpdateHeadPoseJumpMonitor();

        if (logStableStatus && Time.unscaledTime >= _nextStatusLogAt)
        {
            _nextStatusLogAt = Time.unscaledTime + Mathf.Max(0.5f, statusLogIntervalSeconds);
            LogStatus("Status");
        }
    }

    private bool ShouldRunInCurrentRuntime()
    {
        if (!runInEditor && Application.isEditor)
            return false;

#if UNITY_ANDROID
        return true;
#else
        return !androidOnly;
#endif
    }

    private void OnHMDAcquired()
    {
        OnRuntimeStateRecovered("HMDAcquired");
    }

    private void OnVrFocusAcquired()
    {
        OnRuntimeStateRecovered("VrFocusAcquired");
    }

    private void OnInputFocusAcquired()
    {
        OnRuntimeStateRecovered("InputFocusAcquired");
    }

    private void OnTrackingAcquired()
    {
        OnRuntimeStateRecovered("TrackingAcquired");
    }

    private void OnHMDLost()
    {
        OnRuntimeStateLost("HMDLost");
    }

    private void OnVrFocusLost()
    {
        OnRuntimeStateLost("VrFocusLost");
    }

    private void OnInputFocusLost()
    {
        OnRuntimeStateLost("InputFocusLost");
    }

    private void OnTrackingLost()
    {
        OnRuntimeStateLost("TrackingLost");
    }

    private void OnRuntimeStateRecovered(string reason)
    {
        if (!ShouldRunInCurrentRuntime())
            return;

        float now = Time.unscaledTime;
        if (string.Equals(reason, _lastRecoveryReason, StringComparison.Ordinal)
            && (now - _lastRecoverySignalAt) < EventDedupeSeconds)
        {
            return;
        }

        _lastRecoveryReason = reason;
        _lastRecoverySignalAt = now;
        if (_runtimeRecoveryPending || !_runtimeLossActive)
        {
            EnforceSettings("RuntimeEvent");
            return;
        }

        _runtimeLossActive = false;
        _runtimeRecoveryPending = true;
        LastRecoveryAt = now;
        _stableAfterTime = now + Mathf.Max(0.0f, settleWindowSeconds);
        _hasLastHeadWorldPosition = false;
        ResetWorldFrameBaseline("XR recovery");
        Debug.Log(
            $"[QuestTrackingOriginGuard] XR recovery observed ({reason}) "
            + $"stableAfter={_stableAfterTime:F2} settleWindow={Mathf.Max(0.0f, settleWindowSeconds):F2}s"
        );
        EnforceSettings("RuntimeEvent", logWhenAlreadyCorrect: true);
    }

    private void OnRuntimeStateLost(string reason)
    {
        if (!ShouldRunInCurrentRuntime())
            return;

        float now = Time.unscaledTime;
        if (string.Equals(reason, _lastLossReason, StringComparison.Ordinal)
            && (now - _lastLossSignalAt) < EventDedupeSeconds)
        {
            return;
        }

        _lastLossReason = reason;
        _lastLossSignalAt = now;
        _runtimeLossActive = true;
        _runtimeRecoveryPending = false;
        _stableAfterTime = now;
        LastUnstableAt = now;
        _hasLastHeadWorldPosition = false;
        ResetWorldFrameBaseline("XR loss");
        Debug.LogWarning(
            $"[QuestTrackingOriginGuard] XR loss observed ({reason}) "
            + $"lastRecoveryAt={LastRecoveryAt:F2}"
        );
    }

    private void UpdateHeadPoseJumpMonitor()
    {
        Transform head = ResolveHeadTransform();
        if (head == null)
        {
            _headTransform = null;
            _hasLastHeadWorldPosition = false;
            return;
        }

        if (_headTransform != head)
        {
            _headTransform = head;
            _lastHeadWorldPosition = head.position;
            _hasLastHeadWorldPosition = true;
            return;
        }

        Vector3 currentHeadWorldPos = head.position;
        if (_hasLastHeadWorldPosition)
        {
            // Intentionally keep head-jump monitoring diagnostic-only for this pass.
            // Raw world-space jumps during app-space churn must not affect runtime stability.
            Vector3.Distance(currentHeadWorldPos, _lastHeadWorldPosition);
        }

        _lastHeadWorldPosition = currentHeadWorldPos;
        _hasLastHeadWorldPosition = true;
    }

    private Transform ResolveHeadTransform()
    {
        if (_headTransform != null)
            return _headTransform;

        if (!string.IsNullOrWhiteSpace(preferredHeadTransformName))
        {
            var allTransforms = FindObjectsByType<Transform>(
                FindObjectsInactive.Exclude,
                FindObjectsSortMode.None
            );
            for (int i = 0; i < allTransforms.Length; i++)
            {
                var current = allTransforms[i];
                if (current == null)
                    continue;
                if (string.Equals(current.name, preferredHeadTransformName))
                {
                    _headTransform = current;
                    return _headTransform;
                }
            }
        }

        if (Camera.main != null)
        {
            _headTransform = Camera.main.transform;
            return _headTransform;
        }

        var anyCamera = FindAnyObjectByType<Camera>();
        if (anyCamera != null)
        {
            _headTransform = anyCamera.transform;
            return _headTransform;
        }

        if (OVRManager.instance != null)
            _headTransform = OVRManager.instance.transform;
        return _headTransform;
    }

    private Transform ResolveRigRootTransform()
    {
        if (_rigRootTransform != null)
            return _rigRootTransform;

        _rigRootTransform = FindTransformByName(preferredRigRootName, includeInactive: true);
        if (_rigRootTransform != null)
            return _rigRootTransform;

        var cameraRig = FindAnyObjectByType<OVRCameraRig>();
        if (cameraRig != null)
        {
            _rigRootTransform = cameraRig.transform;
            return _rigRootTransform;
        }

        var manager = ResolveManager();
        if (manager != null)
        {
            _rigRootTransform = manager.transform;
            return _rigRootTransform;
        }
        return _rigRootTransform;
    }

    private Transform ResolveTrackingSpaceTransform()
    {
        if (_trackingSpaceTransform != null)
            return _trackingSpaceTransform;

        var cameraRig = FindAnyObjectByType<OVRCameraRig>();
        if (cameraRig != null && cameraRig.trackingSpace != null)
        {
            _trackingSpaceTransform = cameraRig.trackingSpace;
            return _trackingSpaceTransform;
        }

        var rigRoot = ResolveRigRootTransform();
        if (rigRoot != null)
            _trackingSpaceTransform = FindDescendantByName(rigRoot, preferredTrackingSpaceName, includeInactive: true);
        else
            _trackingSpaceTransform = FindTransformByName(preferredTrackingSpaceName, includeInactive: true);
        return _trackingSpaceTransform;
    }

    private void EnsureLocomotorDisabled()
    {
        if (!disableLocomotorOnStart)
            return;

        var rigRoot = ResolveRigRootTransform();
        if (rigRoot == null)
            return;

        var locomotor = FindDescendantByName(rigRoot, locomotorObjectName, includeInactive: true);
        if (locomotor == null)
        {
            if (!_loggedLocomotorStatus)
            {
                Debug.LogWarning(
                    $"[QuestTrackingOriginGuard] Could not find locomotion object '{locomotorObjectName}' under '{rigRoot.name}'."
                );
                _loggedLocomotorStatus = true;
            }
            return;
        }

        if (locomotor.gameObject.activeSelf)
        {
            locomotor.gameObject.SetActive(false);
            Debug.LogWarning(
                $"[QuestTrackingOriginGuard] Disabled locomotion object '{locomotor.name}' under '{rigRoot.name}' to prevent camera-rig world drift."
            );
            _loggedLocomotorStatus = true;
            ResetWorldFrameBaseline("Locomotor disabled");
            return;
        }

        if (!_loggedLocomotorStatus)
        {
            Debug.Log(
                $"[QuestTrackingOriginGuard] Verified locomotion object '{locomotor.name}' is disabled for this scene."
            );
            _loggedLocomotorStatus = true;
        }
    }

    private void ResetWorldFrameBaseline(string reason)
    {
        _hasWorldFrameBaseline = false;
        _worldFrameDriftActive = false;
        _currentRigRootWorldDelta = Vector3.zero;
        _currentTrackingSpaceWorldDelta = Vector3.zero;
        _nextWorldFrameLogAt = float.NegativeInfinity;
        if (!string.IsNullOrWhiteSpace(reason))
        {
            Debug.Log(
                $"[QuestTrackingOriginGuard] Reset world-frame baseline ({reason})."
            );
        }
    }

    private void UpdateWorldFrameState(string reason)
    {
        var rigRoot = ResolveRigRootTransform();
        var trackingSpace = ResolveTrackingSpaceTransform();
        if (rigRoot == null || trackingSpace == null)
        {
            if (Time.unscaledTime >= _nextWorldFrameLogAt)
            {
                _nextWorldFrameLogAt = Time.unscaledTime + Mathf.Max(0.25f, worldFrameLogIntervalSeconds);
                Debug.LogWarning(
                    "[QuestTrackingOriginGuard] World-frame monitor is waiting for XR rig transforms. "
                    + $"rigRoot={DescribeTransform(rigRoot)} trackingSpace={DescribeTransform(trackingSpace)}"
                );
            }
            return;
        }

        if (!IsRuntimeStable)
            return;

        if (!_hasWorldFrameBaseline)
        {
            _baselineRigRootWorldPosition = rigRoot.position;
            _baselineRigRootWorldRotation = rigRoot.rotation;
            _baselineTrackingSpaceWorldPosition = trackingSpace.position;
            _currentRigRootWorldDelta = Vector3.zero;
            _currentTrackingSpaceWorldDelta = Vector3.zero;
            _worldFrameDriftActive = false;
            _hasWorldFrameBaseline = true;
            LastWorldFrameBaselineAt = Time.unscaledTime;
            Debug.Log(
                "[QuestTrackingOriginGuard] Captured rig world-frame baseline "
                + $"reason='{reason}' rigRoot={DescribeTransform(rigRoot)} "
                + $"trackingSpace={DescribeTransform(trackingSpace)}"
            );
            _nextWorldFrameLogAt = Time.unscaledTime + Mathf.Max(0.25f, worldFrameLogIntervalSeconds);
            return;
        }

        _currentRigRootWorldDelta = rigRoot.position - _baselineRigRootWorldPosition;
        _currentTrackingSpaceWorldDelta = trackingSpace.position - _baselineTrackingSpaceWorldPosition;

        bool rigRootDrifting =
            _currentRigRootWorldDelta.magnitude > Mathf.Max(0.001f, maxRigRootWorldDriftMeters);
        bool trackingSpaceDrifting =
            _currentTrackingSpaceWorldDelta.magnitude > Mathf.Max(0.001f, maxTrackingSpaceWorldDriftMeters);
        bool driftNow = rigRootDrifting || trackingSpaceDrifting;

        if (driftNow && snapRigRootToBaselineWhenDriftPersists
            && _currentRigRootWorldDelta.magnitude >= Mathf.Max(0.001f, snapRigRootDriftMeters))
        {
            rigRoot.position = _baselineRigRootWorldPosition;
            rigRoot.rotation = _baselineRigRootWorldRotation;
            _currentRigRootWorldDelta = rigRoot.position - _baselineRigRootWorldPosition;
            _currentTrackingSpaceWorldDelta = trackingSpace.position - _baselineTrackingSpaceWorldPosition;
            rigRootDrifting =
                _currentRigRootWorldDelta.magnitude > Mathf.Max(0.001f, maxRigRootWorldDriftMeters);
            trackingSpaceDrifting =
                _currentTrackingSpaceWorldDelta.magnitude > Mathf.Max(0.001f, maxTrackingSpaceWorldDriftMeters);
            driftNow = rigRootDrifting || trackingSpaceDrifting;
            Debug.LogWarning(
                "[QuestTrackingOriginGuard] Snapped camera-rig root back to the captured baseline "
                + $"reason='{reason}' rigRootDelta={FormatVector(_currentRigRootWorldDelta)} "
                + $"trackingSpaceDelta={FormatVector(_currentTrackingSpaceWorldDelta)}"
            );
        }

        if (driftNow)
            LastWorldFrameDriftAt = Time.unscaledTime;

        if (driftNow != _worldFrameDriftActive || Time.unscaledTime >= _nextWorldFrameLogAt)
        {
            _nextWorldFrameLogAt = Time.unscaledTime + Mathf.Max(0.25f, worldFrameLogIntervalSeconds);
            Debug.Log(
                "[QuestTrackingOriginGuard] World-frame state "
                + $"reason='{reason}' stable={!driftNow} "
                + $"rigRootDelta={FormatVector(_currentRigRootWorldDelta)} "
                + $"trackingSpaceDelta={FormatVector(_currentTrackingSpaceWorldDelta)} "
                + $"baselineRigRootPos={_baselineRigRootWorldPosition} "
                + $"baselineTrackingSpacePos={_baselineTrackingSpaceWorldPosition}"
            );
        }

        _worldFrameDriftActive = driftNow;
    }

    private void EnforceSettings(string reason, bool logWhenAlreadyCorrect = false)
    {
        var manager = ResolveManager();
        if (manager == null)
            return;

        var currentOrigin = manager.trackingOriginType;
        bool currentAllowRecenter = manager.AllowRecenter;
        bool originNeedsUpdate = currentOrigin != desiredTrackingOrigin;
        bool recenterNeedsUpdate = currentAllowRecenter != desiredAllowRecenter;

        if (!originNeedsUpdate && !recenterNeedsUpdate)
        {
            if (logWhenAlreadyCorrect)
                LogStatus(reason);
            return;
        }

        if (recenterNeedsUpdate)
            manager.AllowRecenter = desiredAllowRecenter;
        if (originNeedsUpdate)
            manager.trackingOriginType = desiredTrackingOrigin;

        _reapplyCount++;
        Debug.LogWarning(
            $"[QuestTrackingOriginGuard] Reapplied XR state ({reason}) "
            + $"trackingOrigin={currentOrigin}->{manager.trackingOriginType} "
            + $"allowRecenter={currentAllowRecenter}->{manager.AllowRecenter} "
            + $"reapplyCount={_reapplyCount}"
        );
    }

    private void LogStatus(string reason)
    {
        var manager = ResolveManager();
        if (manager == null)
            return;

        float settleRemaining = Mathf.Max(0.0f, _stableAfterTime - Time.unscaledTime);
        Debug.Log(
            $"[QuestTrackingOriginGuard] {reason}: trackingOrigin={manager.trackingOriginType} "
            + $"allowRecenter={manager.AllowRecenter} desiredTrackingOrigin={desiredTrackingOrigin} "
            + $"desiredAllowRecenter={desiredAllowRecenter} reapplyCount={_reapplyCount} "
            + $"runtimeStable={IsRuntimeStable} worldFrameStable={IsWorldFrameStable} "
            + $"hasWorldFrameBaseline={_hasWorldFrameBaseline} "
            + $"rigRootDelta={FormatVector(_currentRigRootWorldDelta)} "
            + $"trackingSpaceDelta={FormatVector(_currentTrackingSpaceWorldDelta)} "
            + $"settleRemaining={settleRemaining:F2}s "
            + $"lastUnstableAt={LastUnstableAt:F2} lastRecoveryAt={LastRecoveryAt:F2} "
            + $"lastWorldFrameBaselineAt={LastWorldFrameBaselineAt:F2} lastWorldFrameDriftAt={LastWorldFrameDriftAt:F2}"
        );
    }

    private static OVRManager ResolveManager()
    {
        var manager = OVRManager.instance;
        if (manager == null)
            manager = FindAnyObjectByType<OVRManager>();
        return manager;
    }

    private static Transform FindTransformByName(string transformName, bool includeInactive)
    {
        if (string.IsNullOrWhiteSpace(transformName))
            return null;

        var allTransforms = FindObjectsByType<Transform>(
            includeInactive ? FindObjectsInactive.Include : FindObjectsInactive.Exclude,
            FindObjectsSortMode.None
        );
        for (int i = 0; i < allTransforms.Length; i++)
        {
            var current = allTransforms[i];
            if (current == null)
                continue;
            if (string.Equals(current.name, transformName, StringComparison.Ordinal))
                return current;
        }

        return null;
    }

    private static Transform FindDescendantByName(Transform root, string transformName, bool includeInactive)
    {
        if (root == null || string.IsNullOrWhiteSpace(transformName))
            return null;

        var children = root.GetComponentsInChildren<Transform>(includeInactive);
        for (int i = 0; i < children.Length; i++)
        {
            var current = children[i];
            if (current == null)
                continue;
            if (string.Equals(current.name, transformName, StringComparison.Ordinal))
                return current;
        }

        return null;
    }

    private static string DescribeTransform(Transform tf)
    {
        if (tf == null)
            return "<none>";

        return $"'{tf.name}' path='{GetTransformPath(tf)}' localPos={tf.localPosition} worldPos={tf.position}";
    }

    private static string FormatVector(Vector3 vector)
    {
        return $"({vector.x:F3}, {vector.y:F3}, {vector.z:F3})";
    }

    private static string GetTransformPath(Transform tf)
    {
        if (tf == null)
            return "<none>";

        var names = new System.Collections.Generic.List<string>(8);
        for (Transform current = tf; current != null; current = current.parent)
            names.Add(current.name);
        names.Reverse();
        return string.Join("/", names);
    }
}
