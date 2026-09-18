using System;
using UnityEngine;
using IRIS.SceneLoader;

/// <summary>
/// Keeps the spawned Mujoco scene in front of the headset once it appears,
/// and re-applies alignment after XR recenter events.
/// </summary>
public class MujocoSceneAutoAlign : MonoBehaviour
{
    public string sceneName = "MujocoScene";
    public float distance = 0.3f;
    public float verticalOffset = -1.0f;
    public bool alignYawToHead = true;
    public bool keepWatchingForRecenter = true;
    [Header("XR Frame")]
    public bool anchorToTrackingSpace = true;
    public string preferredTrackingSpaceName = "TrackingSpace";
    [Header("Startup Stabilization")]
    public bool requireStableTrackingBeforeAlignment = true;
    public int stableChecksBeforeAlignment = 3;
    public bool freezeAfterFirstStableAlignment = true;
    public float transformDiagnosticsLogInterval = 2.0f;
    [Header("Fixed Table Height (floor-anchored)")]
    // When true, ignores head height and places the scene so the virtual table top
    // sits at exactly fixedTableTopHeight metres above the tracking-space floor (Y=0).
    // Requires the Quest tracking origin to be set to Floor Level.
    // MuJoCo table top is at Z=0 → Unity Y=0 relative to scene root, so scene root Y = fixedTableTopHeight.
    public bool useFixedTableHeight = true;
    public float fixedTableTopHeight = 1.15f;
    [Header("Scene Scale")]
    // Uniform multiplier applied to the spawned MuJoCo scene root.
    // 1.0 = authored scale, >1.0 bigger in VR.
    public float sceneScale = 1.10f;
    [Header("Safety")]
    public bool preserveSceneHeight = true;
    public float minReasonableHeadY = -2.0f;
    public float maxReasonableHeadY = 3.5f;
    public bool keepWatchingForAbnormalHeadPose = true;
    public bool useHeadRelativeFallbackWhenHeadYIsAbnormal = false;
    public float maxHeadPoseJumpMeters = 1.0f;

    private Transform _lastSceneRoot;
    private Transform _lastAlignmentFrame;
    private Transform _lastReparentedSceneRoot;
    private bool? _lastHeadYWasReasonable;
    private bool _hasLastAlignedHeadPose;
    private Vector3 _lastAlignedHeadPosInAlignmentFrame;
    private float _lastObservedRecoveryAt = float.NegativeInfinity;
    private Transform _lastSceneSnapshotRoot;
    private QuestTrackingOriginGuard _trackingOriginGuard;
    private Transform _originalSceneParent;
    private bool _hasCompletedStableAlignment;
    private int _stableChecks;
    private Vector3 _sceneBaseLocalScale = Vector3.one;
    private float _nextTransformDiagnosticsLogAt = float.NegativeInfinity;
    private float _nextGateLogAt = float.NegativeInfinity;

    private const float AlignmentGateLogIntervalSeconds = 2.0f;
    private bool _loggedCombinedViewDisabled;

    void Update()
    {
        // Combined view has a different owner for MujocoScene: SceneAnchorManager restores
        // and persists the user-tuned PlayerPrefs pose. This legacy auto-aligner otherwise
        // reparents the same root under TrackingSpace and writes a head-relative position/
        // yaw after the saved pose has been restored, which makes the preference appear to
        // drift whenever the additive intervention scene is entered or rebuilt.
        if (MultiSessionGridManager.CombinedViewEnabled)
        {
            if (!_loggedCombinedViewDisabled)
            {
                _loggedCombinedViewDisabled = true;
                Debug.Log("[MujocoSceneAutoAlign] Combined view active; disabled because "
                          + "SceneAnchorManager owns the persisted MujocoScene pose.");
            }
            return;
        }

        if (SimSceneSpawner.Instance == null)
            return;

        var root = SimSceneSpawner.Instance.GetSceneTransform(sceneName);
        if (root == null)
            return;

        var cam = Camera.main;
        Transform head = cam != null ? cam.transform : null;
        var alignmentFrame = ResolveAlignmentFrame(head);
        var trackingOriginGuard = ResolveTrackingOriginGuard();

        bool sceneChanged = _lastSceneRoot != root;
        bool alignmentFrameChanged = _lastAlignmentFrame != alignmentFrame;
        if (sceneChanged)
            HandleSceneRootChanged(root);

        bool recoveryChanged = false;
        if (keepWatchingForRecenter
            && trackingOriginGuard != null
            && trackingOriginGuard.LastRecoveryAt > _lastObservedRecoveryAt)
        {
            _lastObservedRecoveryAt = trackingOriginGuard.LastRecoveryAt;
            recoveryChanged = trackingOriginGuard.LastRecoveryAt > float.NegativeInfinity;
        }

        if (recoveryChanged)
        {
            _hasCompletedStableAlignment = false;
            _hasLastAlignedHeadPose = false;
            _stableChecks = 0;
            RestoreOriginalParent(
                root,
                "XR recovery observed; waiting for stable tracking before re-aligning the scene."
            );
        }

        _lastSceneRoot = root;
        _lastAlignmentFrame = alignmentFrame;

        if (head == null)
        {
            LogAlignmentGateBlocked(
                root,
                alignmentFrame,
                trackingOriginGuard,
                runtimeStable: trackingOriginGuard == null || trackingOriginGuard.IsRuntimeStable,
                worldFrameStable: trackingOriginGuard == null || trackingOriginGuard.IsWorldFrameStable,
                headAvailable: false,
                headYInAlignmentFrame: float.NaN,
                headYInAlignmentFrameReasonable: false
            );
            LogTransformChainDiagnostics(head, alignmentFrame, "camera-missing");
            return;
        }

        float headYInAlignmentFrame = GetHeadPositionInAlignmentFrame(head, alignmentFrame).y;
        bool headYInAlignmentFrameReasonable = IsHeadYReasonable(headYInAlignmentFrame);
        _lastHeadYWasReasonable = headYInAlignmentFrameReasonable;

        bool runtimeStable = !requireStableTrackingBeforeAlignment
            || trackingOriginGuard == null
            || trackingOriginGuard.IsRuntimeStable;
        bool worldFrameStable = !requireStableTrackingBeforeAlignment
            || trackingOriginGuard == null
            || trackingOriginGuard.IsWorldFrameStable;
        bool headPoseAcceptable = !keepWatchingForAbnormalHeadPose
            || headYInAlignmentFrameReasonable;

        if (runtimeStable && worldFrameStable && headPoseAcceptable)
            _stableChecks = Mathf.Min(_stableChecks + 1, 1024);
        else
            _stableChecks = 0;

        bool stableGatePassed = _stableChecks >= Mathf.Max(1, stableChecksBeforeAlignment);
        if (!_hasCompletedStableAlignment && !stableGatePassed)
        {
            LogAlignmentGateBlocked(
                root,
                alignmentFrame,
                trackingOriginGuard,
                runtimeStable,
                worldFrameStable,
                headAvailable: true,
                headYInAlignmentFrame,
                headYInAlignmentFrameReasonable
            );
            LogTransformChainDiagnostics(head, alignmentFrame, "waiting-for-stable-frame");
            return;
        }

        if (freezeAfterFirstStableAlignment && _hasCompletedStableAlignment && !sceneChanged && !recoveryChanged)
            return;

        bool aligned = AlignSceneRoot(root, alignmentFrame, head);
        if (!aligned)
            return;

        _hasCompletedStableAlignment = true;
        if (_lastSceneSnapshotRoot != root || recoveryChanged || sceneChanged || alignmentFrameChanged)
        {
            LogSceneSnapshot(root);
            _lastSceneSnapshotRoot = root;
        }
    }

    private bool IsHeadYReasonable(float headY)
    {
        return headY >= minReasonableHeadY && headY <= maxReasonableHeadY;
    }

    private Transform ResolveAlignmentFrame(Transform head)
    {
        if (!anchorToTrackingSpace || head == null)
            return null;

        for (Transform current = head; current != null; current = current.parent)
        {
            if (string.Equals(current.name, preferredTrackingSpaceName, StringComparison.Ordinal))
                return current;
        }

        return head.parent;
    }

    private Vector3 GetHeadPositionInAlignmentFrame(Transform head, Transform alignmentFrame)
    {
        if (head == null)
            return Vector3.zero;
        return alignmentFrame != null
            ? alignmentFrame.InverseTransformPoint(head.position)
            : head.position;
    }

    private Vector3 GetHeadForwardFlatInAlignmentFrame(Transform head, Transform alignmentFrame)
    {
        if (head == null)
            return Vector3.forward;

        Vector3 forwardFlat = alignmentFrame != null
            ? alignmentFrame.InverseTransformDirection(head.forward)
            : head.forward;
        forwardFlat.y = 0f;
        if (forwardFlat.sqrMagnitude < 1e-6f)
            forwardFlat = Vector3.forward;
        return forwardFlat.normalized;
    }

    private void HandleSceneRootChanged(Transform root)
    {
        _originalSceneParent = root != null ? root.parent : null;
        _sceneBaseLocalScale = root != null ? root.localScale : Vector3.one;
        _lastReparentedSceneRoot = null;
        _lastSceneSnapshotRoot = null;
        _hasCompletedStableAlignment = false;
        _hasLastAlignedHeadPose = false;
        _stableChecks = 0;
    }

    private bool AlignSceneRoot(Transform root, Transform alignmentFrame, Transform head)
    {
        if (head == null)
            return false;

        bool useAlignmentFrame = alignmentFrame != null;
        if (useAlignmentFrame && _lastReparentedSceneRoot != root)
        {
            if (root.parent != alignmentFrame)
            {
                root.SetParent(alignmentFrame, true);
                Debug.Log(
                    $"[MujocoSceneAutoAlign] Re-parented '{sceneName}' under alignment frame '{alignmentFrame.name}'."
                );
            }
            _lastReparentedSceneRoot = root;
        }

        Vector3 headPos = GetHeadPositionInAlignmentFrame(head, alignmentFrame);
        Vector3 forwardFlat = GetHeadForwardFlatInAlignmentFrame(head, alignmentFrame);
        if (_hasLastAlignedHeadPose)
        {
            float headJump = Vector3.Distance(_lastAlignedHeadPosInAlignmentFrame, headPos);
            if (headJump > Mathf.Max(0.01f, maxHeadPoseJumpMeters))
            {
                Debug.LogWarning(
                    $"[MujocoSceneAutoAlign] Head pose jumped by {headJump:F2}m for '{sceneName}'. "
                    + $"Preserving current scene pose until tracking stabilizes."
                );
                _lastAlignedHeadPosInAlignmentFrame = headPos;
                return false;
            }
        }

        bool headYIsReasonable = IsHeadYReasonable(headPos.y);
        // Only abort on bad head Y when using head-relative height — fixed table height doesn't need it.
        if (!useFixedTableHeight && !headYIsReasonable && !useHeadRelativeFallbackWhenHeadYIsAbnormal)
        {
            Vector3 currentPos = useAlignmentFrame ? root.localPosition : root.position;
            Debug.LogWarning(
                $"[MujocoSceneAutoAlign] Head y={headPos.y:F2} is outside "
                + $"[{minReasonableHeadY:F2}, {maxReasonableHeadY:F2}] for '{sceneName}'. "
                + $"Preserving current scene pose at pos={currentPos} frame='{(useAlignmentFrame ? alignmentFrame.name : "world")}'."
            );
            return false;
        }

        float safeSceneScale = Mathf.Clamp(sceneScale, 0.5f, 2.0f);
        Vector3 targetScale = _sceneBaseLocalScale * safeSceneScale;
        if ((root.localScale - targetScale).sqrMagnitude > 1e-6f)
            root.localScale = targetScale;

        Vector3 targetPos = headPos + (forwardFlat * distance);
        float targetY;
        if (useFixedTableHeight)
        {
            // Floor-anchored: scene root Y = table top height above tracking-space floor.
            // Quest 3 tracking origin must be set to Floor Level (Y=0 = floor).
            // MuJoCo table top at Z=0 → Unity Y=0 relative to scene root, so this is exact.
            targetY = fixedTableTopHeight;
        }
        else if (headYIsReasonable || !preserveSceneHeight || useHeadRelativeFallbackWhenHeadYIsAbnormal)
        {
            targetY = headPos.y + verticalOffset;
            if (!headYIsReasonable)
            {
                Debug.LogWarning(
                    $"[MujocoSceneAutoAlign] Head y={headPos.y:F2} is outside "
                    + $"[{minReasonableHeadY:F2}, {maxReasonableHeadY:F2}] for '{sceneName}'. "
                    + $"Using head-relative fallback inside frame='{(useAlignmentFrame ? alignmentFrame.name : "world")}'."
                );
            }
        }
        else
        {
            targetY = useAlignmentFrame ? root.localPosition.y : root.position.y;
            Debug.LogWarning(
                $"[MujocoSceneAutoAlign] Head y={headPos.y:F2} is outside "
                + $"[{minReasonableHeadY:F2}, {maxReasonableHeadY:F2}] for '{sceneName}'. "
                + $"Keeping current scene y={targetY:F2}."
            );
        }

        targetPos.y = targetY;
        if (useAlignmentFrame)
            root.localPosition = targetPos;
        else
            root.position = targetPos;

        if (alignYawToHead)
        {
            Quaternion targetRot = Quaternion.LookRotation(-forwardFlat, Vector3.up);
            if (useAlignmentFrame)
                root.localRotation = targetRot;
            else
                root.rotation = targetRot;
        }

        Vector3 appliedPos = useAlignmentFrame ? root.localPosition : root.position;
        float appliedYaw = useAlignmentFrame ? root.localEulerAngles.y : root.eulerAngles.y;
        _lastAlignedHeadPosInAlignmentFrame = headPos;
        _hasLastAlignedHeadPose = true;
        Debug.Log(
            $"[MujocoSceneAutoAlign] Aligned '{sceneName}' at pos={appliedPos} "
            + $"yaw={appliedYaw:F1} preserveSceneHeight={preserveSceneHeight} "
            + $"headYReasonable={headYIsReasonable} frame='{(useAlignmentFrame ? alignmentFrame.name : "world")}' "
            + $"stableChecks={_stableChecks}/{Mathf.Max(1, stableChecksBeforeAlignment)} "
            + $"sceneScale={safeSceneScale:F2}"
        );
        LogTransformChainDiagnostics(head, alignmentFrame, "post-align");
        return true;
    }

    private void RestoreOriginalParent(Transform root, string reason)
    {
        if (root == null)
            return;

        if (root.parent == _originalSceneParent)
            return;

        root.SetParent(_originalSceneParent, true);
        _lastReparentedSceneRoot = null;
        string parentLabel = _originalSceneParent != null ? _originalSceneParent.name : "<world>";
        Debug.LogWarning(
            $"[MujocoSceneAutoAlign] Restored '{sceneName}' under original parent '{parentLabel}'. {reason}"
        );
    }

    private void LogAlignmentGateBlocked(
        Transform root,
        Transform alignmentFrame,
        QuestTrackingOriginGuard trackingOriginGuard,
        bool runtimeStable,
        bool worldFrameStable,
        bool headAvailable,
        float headYInAlignmentFrame,
        bool headYInAlignmentFrameReasonable
    )
    {
        if (Time.unscaledTime < _nextGateLogAt)
            return;

        _nextGateLogAt = Time.unscaledTime + AlignmentGateLogIntervalSeconds;
        string rootPath = GetTransformPath(root);
        string alignmentFrameName = alignmentFrame != null ? alignmentFrame.name : "world";
        string rigRootDelta = trackingOriginGuard != null
            ? trackingOriginGuard.CurrentRigRootWorldDelta.ToString("F3")
            : "n/a";
        string trackingSpaceDelta = trackingOriginGuard != null
            ? trackingOriginGuard.CurrentTrackingSpaceWorldDelta.ToString("F3")
            : "n/a";
        bool hasWorldFrameBaseline = trackingOriginGuard == null || trackingOriginGuard.HasWorldFrameBaseline;
        Debug.LogWarning(
            $"[MujocoSceneAutoAlign] Waiting for stable XR frame before aligning '{sceneName}'. "
            + $"stableChecks={_stableChecks}/{Mathf.Max(1, stableChecksBeforeAlignment)} "
            + $"runtimeStable={runtimeStable} headAvailable={headAvailable} "
            + $"worldFrameStable={worldFrameStable} hasWorldFrameBaseline={hasWorldFrameBaseline} "
            + $"headYAlignment={(headAvailable ? headYInAlignmentFrame.ToString("F2") : "n/a")} "
            + $"headYAlignmentReasonable={headYInAlignmentFrameReasonable} "
            + $"rigRootDelta={rigRootDelta} trackingSpaceDelta={trackingSpaceDelta} "
            + $"alignmentFrame='{alignmentFrameName}' "
            + $"scenePath='{rootPath}'"
        );
    }

    private void LogTransformChainDiagnostics(Transform head, Transform alignmentFrame, string reason)
    {
        if (Time.unscaledTime < _nextTransformDiagnosticsLogAt)
            return;

        _nextTransformDiagnosticsLogAt = Time.unscaledTime + Mathf.Max(0.25f, transformDiagnosticsLogInterval);
        Transform rigParent = alignmentFrame != null ? alignmentFrame.parent : (head != null ? head.parent : null);
        Debug.Log(
            "[MujocoSceneAutoAlign] Transform diagnostics: "
            + $"reason='{reason}' "
            + $"head={DescribeTransform(head)} "
            + $"headParent={DescribeTransform(head != null ? head.parent : null)} "
            + $"alignmentFrame={DescribeTransform(alignmentFrame)} "
            + $"rigParent={DescribeTransform(rigParent)} "
            + $"sceneParent={DescribeTransform(_lastSceneRoot != null ? _lastSceneRoot.parent : null)} "
            + $"originalSceneParent={DescribeTransform(_originalSceneParent)}"
        );
    }

    private void LogSceneSnapshot(Transform root)
    {
        if (root == null)
            return;

        Transform pointCloudAnchor = FindFirstChildWithPrefix(root, "PointCloudAnchor_");
        Transform panelAnchor = FindFirstChildByName(root, "hand")
            ?? FindFirstChildByName(root, "link7");
        Debug.Log(
            $"[MujocoSceneAutoAlign] Scene snapshot path='{GetTransformPath(root)}' "
            + $"localPos={root.localPosition} localRot={root.localEulerAngles} localScale={root.localScale} "
            + $"worldPos={root.position} worldRot={root.rotation.eulerAngles} worldScale={root.lossyScale} "
            + $"pointCloudAnchorResolved={(pointCloudAnchor != null)} "
            + $"rgbPanelAnchorResolved={(panelAnchor != null)}"
        );
    }

    private QuestTrackingOriginGuard ResolveTrackingOriginGuard()
    {
        if (_trackingOriginGuard == null)
            _trackingOriginGuard = QuestTrackingOriginGuard.ResolveSharedGuard();
        return _trackingOriginGuard;
    }

    private static Transform FindFirstChildWithPrefix(Transform root, string prefix)
    {
        if (root == null || string.IsNullOrWhiteSpace(prefix))
            return null;

        var allChildren = root.GetComponentsInChildren<Transform>(true);
        for (int i = 0; i < allChildren.Length; i++)
        {
            var child = allChildren[i];
            if (child == null)
                continue;
            if (child.name.StartsWith(prefix, StringComparison.Ordinal))
                return child;
        }
        return null;
    }

    private static Transform FindFirstChildByName(Transform root, string name)
    {
        if (root == null || string.IsNullOrWhiteSpace(name))
            return null;

        var allChildren = root.GetComponentsInChildren<Transform>(true);
        for (int i = 0; i < allChildren.Length; i++)
        {
            var child = allChildren[i];
            if (child == null)
                continue;
            if (string.Equals(child.name, name, StringComparison.Ordinal))
                return child;
        }
        return null;
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

    private static string DescribeTransform(Transform tf)
    {
        if (tf == null)
            return "<none>";

        return $"'{tf.name}' path='{GetTransformPath(tf)}' localPos={tf.localPosition} worldPos={tf.position}";
    }
}
