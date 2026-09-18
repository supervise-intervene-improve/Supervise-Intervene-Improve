using System.Collections;
using UnityEngine;

[DefaultExecutionOrder(-9000)]
public class QuestPassthroughRuntimeGuard : MonoBehaviour
{
    [Header("Runtime Scope")]
    public bool runInEditor = false;
    public bool androidOnly = true;

    [Header("Passthrough Targets")]
    public bool forcePassthroughEnabled = true;
    public bool forceCreateLayerIfMissing = true;
    public bool forceLayerVisible = true;
    public bool forceOverlayType = false;
    public OVROverlay.OverlayType desiredOverlayType = OVROverlay.OverlayType.Underlay;
    public int desiredCompositionDepth = 0;
    [Range(0f, 1f)] public float desiredOpacity = 1f;

    [Header("Underlay Camera")]
    public bool forceTransparentClearForUnderlay = true;

    [Header("Watchdog")]
    public float startupGraceSeconds = 2.5f;
    public float watchdogIntervalSeconds = 1.0f;
    public float statusLogIntervalSeconds = 2.0f;
    public int maxRecoveryAttempts = 12;
    public bool verboseLogs = true;

    [Header("Fallback")]
    public bool disablePassthroughWhenLayerSubmissionFails = true;

    private float _nextStatusLogAt;
    private int _recoveryAttempts;
    private bool _recoveryInFlight;
    private bool _forcedVrFallback;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    private static void Install()
    {
        if (FindAnyObjectByType<QuestPassthroughRuntimeGuard>() != null)
            return;

        var go = new GameObject("[Runtime] QuestPassthroughRuntimeGuard");
        DontDestroyOnLoad(go);
        go.AddComponent<QuestPassthroughRuntimeGuard>();
    }

    private void Start()
    {
        if (!ShouldRunInCurrentRuntime())
        {
            enabled = false;
            return;
        }

        StartCoroutine(WatchdogLoop());
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

    private IEnumerator WatchdogLoop()
    {
        float startTime = Time.unscaledTime;
        while (true)
        {
            EnsurePassthroughConfigured();

            bool hasManager = OVRManager.instance != null;
            bool managerEnabled = hasManager && OVRManager.instance.isInsightPassthroughEnabled;
            bool init = OVRManager.IsInsightPassthroughInitialized();
            bool pending = OVRManager.IsInsightPassthroughInitPending();
            bool failed = OVRManager.HasInsightPassthroughInitFailed();
            CountLayers(out int visibleLayers, out int submittedLayers);

            if (verboseLogs && Time.unscaledTime >= _nextStatusLogAt)
            {
                _nextStatusLogAt = Time.unscaledTime + Mathf.Max(0.5f, statusLogIntervalSeconds);
                Debug.Log(
                    $"[QuestPassthroughRuntimeGuard] manager={hasManager} enabled={managerEnabled} init={init} "
                    + $"pending={pending} failed={failed} visibleLayers={visibleLayers} submittedLayers={submittedLayers} "
                    + $"recoveryAttempts={_recoveryAttempts}"
                );
            }

            bool shouldRecover = (managerEnabled && init && submittedLayers == 0)
                || (managerEnabled && failed);

            if (shouldRecover
                && Time.unscaledTime - startTime >= startupGraceSeconds
                && _recoveryAttempts < Mathf.Max(1, maxRecoveryAttempts)
                && !_recoveryInFlight)
            {
                _recoveryAttempts++;
                StartCoroutine(RecoverPassthroughLayer(_recoveryAttempts));
            }

            if (disablePassthroughWhenLayerSubmissionFails
                && !_forcedVrFallback
                && managerEnabled
                && Time.unscaledTime - startTime >= startupGraceSeconds
                && _recoveryAttempts >= Mathf.Max(1, maxRecoveryAttempts)
                && submittedLayers == 0)
            {
                if (OVRManager.instance != null)
                    OVRManager.instance.isInsightPassthroughEnabled = false;
                _forcedVrFallback = true;
                Debug.LogWarning("[QuestPassthroughRuntimeGuard] Passthrough layer submission failed repeatedly. Forced VR fallback (passthrough OFF).");
            }

            yield return new WaitForSecondsRealtime(Mathf.Max(0.2f, watchdogIntervalSeconds));
        }
    }

    private void EnsurePassthroughConfigured()
    {
        var manager = OVRManager.instance;
        if (manager == null)
            manager = FindAnyObjectByType<OVRManager>();

        if (manager != null && forcePassthroughEnabled && !_forcedVrFallback && !manager.isInsightPassthroughEnabled)
        {
            manager.isInsightPassthroughEnabled = true;
            if (verboseLogs)
                Debug.Log("[QuestPassthroughRuntimeGuard] Forced OVRManager passthrough enabled.");
        }

        var layers = UnityEngine.Object.FindObjectsByType<OVRPassthroughLayer>(
            FindObjectsInactive.Include,
            FindObjectsSortMode.None
        );
        if ((layers == null || layers.Length == 0) && forceCreateLayerIfMissing && manager != null)
        {
            var go = new GameObject("[Runtime] OVRPassthroughLayer");
            go.transform.SetParent(manager.transform, false);

            var layer = go.AddComponent<OVRPassthroughLayer>();
            layer.hidden = false;
            layer.overlayType = desiredOverlayType;
            layer.compositionDepth = desiredCompositionDepth;
            layer.textureOpacity = Mathf.Clamp01(desiredOpacity);

            layers = new[] { layer };
            Debug.LogWarning("[QuestPassthroughRuntimeGuard] No passthrough layer found. Created one at runtime.");
        }

        if (layers == null)
            return;

        for (int i = 0; i < layers.Length; i++)
        {
            var layer = layers[i];
            if (layer == null)
                continue;

            if (!layer.enabled)
                layer.enabled = true;
            if (forceLayerVisible)
                layer.hidden = false;
            if (forceOverlayType)
                layer.overlayType = desiredOverlayType;

            layer.compositionDepth = desiredCompositionDepth;
            layer.textureOpacity = Mathf.Clamp01(desiredOpacity);
        }

        if (forceTransparentClearForUnderlay && HasUnderlayLayer(layers))
            ApplyTransparentCameraClear();
    }

    private static bool HasUnderlayLayer(OVRPassthroughLayer[] layers)
    {
        if (layers == null)
            return false;

        for (int i = 0; i < layers.Length; i++)
        {
            var layer = layers[i];
            if (layer == null || !layer.enabled || layer.hidden)
                continue;
            if (layer.overlayType == OVROverlay.OverlayType.Underlay)
                return true;
        }
        return false;
    }

    private static void ApplyTransparentCameraClear()
    {
        var cams = Camera.allCameras;
        for (int i = 0; i < cams.Length; i++)
        {
            var cam = cams[i];
            if (cam == null || !cam.enabled || cam.targetTexture != null)
                continue;

            cam.clearFlags = CameraClearFlags.SolidColor;
            var bg = cam.backgroundColor;
            if (bg.a != 0f)
            {
                bg.a = 0f;
                cam.backgroundColor = bg;
            }
        }
    }

    private static void CountLayers(out int visibleLayers, out int submittedLayers)
    {
        visibleLayers = 0;
        submittedLayers = 0;

        var layers = UnityEngine.Object.FindObjectsByType<OVRPassthroughLayer>(
            FindObjectsInactive.Include,
            FindObjectsSortMode.None
        );
        if (layers == null)
            return;

        for (int i = 0; i < layers.Length; i++)
        {
            var layer = layers[i];
            if (layer == null || !layer.enabled || layer.hidden)
                continue;

            visibleLayers++;

            var overlays = layer.GetComponentsInChildren<OVROverlay>(true);
            for (int j = 0; j < overlays.Length; j++)
            {
                var ov = overlays[j];
                if (ov != null && ov.enabled && ov.layerId > 0)
                {
                    submittedLayers++;
                    break;
                }
            }
        }
    }

    private IEnumerator RecoverPassthroughLayer(int attempt)
    {
        _recoveryInFlight = true;

        var manager = OVRManager.instance;
        if (manager == null)
            manager = FindAnyObjectByType<OVRManager>();

        if (manager != null)
            manager.isInsightPassthroughEnabled = false;
        yield return null;

        if (manager != null)
            manager.isInsightPassthroughEnabled = true;

        var layers = UnityEngine.Object.FindObjectsByType<OVRPassthroughLayer>(
            FindObjectsInactive.Include,
            FindObjectsSortMode.None
        );
        for (int i = 0; i < layers.Length; i++)
        {
            var layer = layers[i];
            if (layer == null)
                continue;
            layer.enabled = false;
        }
        yield return null;

        for (int i = 0; i < layers.Length; i++)
        {
            var layer = layers[i];
            if (layer == null)
                continue;

            layer.enabled = true;
            if (forceLayerVisible)
                layer.hidden = false;
            if (forceOverlayType)
                layer.overlayType = desiredOverlayType;
            layer.compositionDepth = desiredCompositionDepth;
            layer.textureOpacity = Mathf.Clamp01(desiredOpacity);
        }

        Debug.LogWarning($"[QuestPassthroughRuntimeGuard] Attempted passthrough layer recovery #{attempt}.");
        _recoveryInFlight = false;
    }
}
