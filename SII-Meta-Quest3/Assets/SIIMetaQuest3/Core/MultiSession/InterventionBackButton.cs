using UnityEngine;
using UnityEngine.SceneManagement;

/// <summary>
/// Returns to the session selector from SIIScene_InterventionV1.
/// Pulling the RIGHT index trigger goes back — no visual button needed (suppressed while
/// PC-adjust mode is active, where the left trigger nudges the anchor instead).
/// Only active when a multi-session selection is stored in SessionRegistry (i.e. the
/// operator arrived here from SIIScene_SessionSelector, not from a direct launch).
/// </summary>
public class InterventionBackButton : MonoBehaviour
{
    [Header("Scene")]
    public string selectorSceneName = "SIIScene_SessionSelector";
    [Tooltip("Seconds to let EXIT_SINGLE leave the PUSH socket before loading the selector. "
           + "Kept short: the cmd socket was warmed at hover/enter, so the message leaves "
           + "promptly — this trims the exit-to-grid tail.")]
    public float transitionCommandFlushSeconds = 0.10f;
    [Tooltip("Fallback only. Normal scene switches use local socket cleanup to avoid poisoning new subscribers.")]
    public bool forceGlobalNetMqCleanupOnSceneSwitch = false;

    [Header("Trigger Arming")]
    public float triggerArmingDelaySeconds = 1.5f;
    public bool requireTriggerReleaseBeforeArming = true;

    private bool _transitioning;
    private float _armingDeadline;
    // _armed is read only in the device build (LateUpdate is wrapped in
    // #if !UNITY_EDITOR), so the editor compile sees an assign-but-never-read
    // field. Silence the editor-only CS0414; the field is live on-device.
#pragma warning disable 0414
    private bool _armed;
#pragma warning restore 0414
    private string _ip;
    private int _cmdPort;

    void Start()
    {
        SessionRegistry.Load();
        if (!SessionRegistry.HasSelection)
        {
            gameObject.SetActive(false);
            return;
        }

        _armingDeadline = Time.unscaledTime + Mathf.Max(0f, triggerArmingDelaySeconds);
        _armed = false;
        _ip = SessionRegistry.SelectedPublisherIp;
        _cmdPort = SessionRegistry.SelectedTopicPort + 5;
    }

    void LateUpdate()
    {
        if (_transitioning) return;

#if !UNITY_EDITOR
        if (Time.unscaledTime < _armingDeadline) return;

        if (!_armed)
        {
            if (requireTriggerReleaseBeforeArming
                && OVRInput.Get(OVRInput.RawButton.RIndexTrigger))
                return;
            _armed = true;
        }

        // While adjusting the point-cloud anchor OR the risk chart placement, controller inputs
        // nudge that transform — suppress "exit scene" so a nudge can't bounce to the selector.
        if (SceneAnchorManager.AdjustModeActive || RiskBarChart.AdjustModeActive) return;

        // If the risk bar chart is hovering a bar, that component owns this trigger press
        // (it will call NavigateTo if needed). Do NOT also exit to selector.
        if (RiskBarChart.HoveredBarIndex >= 0) return;

        // Combined view: the grid (in the other, simultaneously-loaded scene) binds the
        // SAME right index trigger to panel-select. If the ray is on a selectable panel,
        // that's a session switch, not an exit — let MultiSessionGridManager.UpdateRay()
        // own this trigger press instead.
        if (MultiSessionGridManager.CombinedViewEnabled && MultiSessionGridManager.IsHoveringSelectablePanel) return;

        // During an intervention the right index trigger is the MC position clutch, and scene
        // exit is a blocked input anyway — the user must ACCEPT (X) or REJECT (grip) the
        // intervention before leaving the scene.
        if (InterventionStatusHud.InterventionActive) return;

        if (OVRInput.GetDown(OVRInput.RawButton.RIndexTrigger))
        {
            _transitioning = true;
            bool exitSent = SessionCommandSender.Send(_ip, _cmdPort, "EXIT_SINGLE");
            SessionRegistry.Clear();
            Debug.Log($"[InterventionBackButton] RightTrigger → EXIT_SINGLE result={exitSent} then returning to {selectorSceneName}");
            StartCoroutine(CleanupAndLoad(selectorSceneName));
        }
#endif
    }

    /// <summary>
    /// Navigate directly to a different session's single view without returning to the selector.
    /// Called by RiskBarChart when the operator clicks on a session's risk bar.
    /// </summary>
    public void NavigateTo(int targetSessionIndex, string ip, int baseTopicPort, int portStep)
    {
        if (_transitioning) return;
        _transitioning = true;

        int targetTopicPort = baseTopicPort + targetSessionIndex * portStep;
        int targetCmdPort   = targetTopicPort + 5;

        // Preserve the current single-view mode. rgbMode is launcher-wide (--rgb_mode), so the
        // target session uses the same mode as the one we're leaving. Hardcoding false here
        // dropped RGB mode into point-cloud mode on every bar-chart jump, so the RGB diamond
        // panels (InterventionRgbPanelSpawner spawns only when SelectedRgbMode) never appeared.
        // Read it BEFORE Save (which overwrites SelectedRgbMode).
        bool targetRgbMode = SessionRegistry.SelectedRgbMode;

        // Exit the current session.
        SessionCommandSender.Send(_ip, _cmdPort, "EXIT_SINGLE");

        // Commit the new session so Bootstrap reconfigures all subscribers on reload.
        SessionRegistry.Save(
            index:       targetSessionIndex,
            topicPort:   targetTopicPort,
            servicePort: targetTopicPort - 1,
            publisherIp: ip,
            label:       $"Session {targetSessionIndex}",
            rgbMode:     targetRgbMode
        );

        // Enter the new session.
        SessionCommandSender.Send(ip, targetCmdPort, "ENTER_SINGLE");
        Debug.Log($"[InterventionBackButton] NavigateTo session {targetSessionIndex} "
                + $"tcp://{ip}:{targetTopicPort} cmd={targetCmdPort}");

        StartCoroutine(CleanupAndLoad("SIIScene_InterventionV1"));
    }

    /// <summary>
    /// Closes local command sockets before unloading the scene.
    /// Global NetMQ cleanup remains available as an explicit fallback, but the
    /// normal path keeps the next scene's fresh subscribers isolated from this
    /// scene's teardown.
    /// </summary>
    private System.Collections.IEnumerator CleanupAndLoad(string sceneName)
    {
        yield return new WaitForSecondsRealtime(Mathf.Max(0.0f, transitionCommandFlushSeconds));

        // Proactively stop our NetMQ subscriber pollers BEFORE unloading/loading, while the
        // shared NetMQ context is still alive. During a scene unload the XR framework node
        // terminates the context in its OnDestroy; if our pollers are
        // still running then, their receive / poller.Stop() throw TerminatingException on the
        // poller thread and abort the app under IL2CPP (observed crash on exit). Each
        // subscriber's OnDisable stops its poller. Scoped to THIS scene only when combined
        // view is active — see StopSceneSubscribers().
        StopSceneSubscribers();

        if (forceGlobalNetMqCleanupOnSceneSwitch)
        {
            try { NetMQ.NetMQConfig.Cleanup(false); } catch { }
        }
        try { SessionCommandSender.Shutdown(); } catch { }
        yield return null;
        yield return null;

        if (MultiSessionGridManager.CombinedViewEnabled)
        {
            if (sceneName == selectorSceneName)
            {
                // Selector is already loaded and untouched — just drop this scene, revealing
                // the still-live grid underneath. No reload needed.
                SceneManager.UnloadSceneAsync(gameObject.scene);
            }
            else
            {
                // NavigateTo() jump to a different session's single view: drop this instance,
                // then load a fresh one additively so InterventionSessionBootstrap's Awake-time
                // rewiring re-runs cleanly for the new session — same mechanism
                // MultiSessionGridManager.ShutdownAndLoad uses for grid-triggered switches.
                //
                // NOTE: this GameObject lives IN the scene being unloaded, so a coroutine
                // hosted on `this` would be destroyed mid-unload before it could reach the
                // LoadSceneAsync line. Chain via the AsyncOperation's completed event instead —
                // that event fires from Unity's scene-loading system, not this coroutine
                // scheduler, so it survives this component being destroyed.
                var unloadOp = SceneManager.UnloadSceneAsync(gameObject.scene);
                string targetScene = sceneName;
                if (unloadOp != null)
                    unloadOp.completed += _ => SceneManager.LoadSceneAsync(targetScene, LoadSceneMode.Additive);
                else
                    SceneManager.LoadSceneAsync(targetScene, LoadSceneMode.Additive);
            }
            yield break;
        }

        // --- Exclusive-swap path (CombinedViewEnabled == false), unchanged ---
        SceneManager.LoadScene(sceneName);
    }

    /// <summary>
    /// Disable every active NetMQ subscriber in the scene so its OnDisable stops the poller while
    /// the shared context is still alive. Mirror of the panel-disable step in
    /// MultiSessionGridManager.ShutdownAndLoad. Safe/idempotent: each subscriber's shutdown is
    /// guarded by its own _shuttingDown flag.
    ///
    /// Combined view: this is an INSTANCE method (not static) here on purpose. Under exclusive
    /// scene swap, SIIScene_InterventionV1 is the only loaded scene, so a scene-global
    /// FindObjectsByType search is equivalent to a scene-scoped one. Under additive loading the
    /// grid scene is ALSO loaded and its 15 panels are SessionThumbnailPanel instances too — a
    /// global FindObjectsByType&lt;SessionThumbnailPanel&gt;() would disable (and kill the NetMQ
    /// subscriber of) every live grid panel the instant the user exits single view. Scope every
    /// lookup to this scene's own root objects when combined view is active.
    /// </summary>
    private void StopSceneSubscribers()
    {
        if (MultiSessionGridManager.CombinedViewEnabled)
        {
            var thisScene = gameObject.scene;
            DisableAllInScene<GpuMergedPointCloudLoader>(thisScene);
            DisableAllInScene<SimPubRgbdSubscriber>(thisScene);
            DisableAllInScene<SimPubPointCloudSubscriber>(thisScene);
            DisableAllInScene<GpuPointCloudSubscriber>(thisScene);
            DisableAllInScene<PointCloudTest>(thisScene);  // class in PointCloudLoader.cs
            DisableAllInScene<SimPubClient>(thisScene);
            DisableAllInScene<InterventionStatusHud>(thisScene);
            // RGB diamond panel mode: the 4 top/left/right/wrist panels spawned by
            // InterventionRgbPanelSpawner are SessionThumbnailPanel instances, each with its
            // own NetMQ poller — same crash class as the others if not stopped proactively.
            // Scoped to this scene ONLY so the grid's 15 panels (other, still-loaded scene)
            // are never touched.
            DisableAllInScene<SessionThumbnailPanel>(thisScene);
            foreach (var rootGo in thisScene.GetRootGameObjects())
            {
                foreach (var chart in rootGo.GetComponentsInChildren<RiskBarChart>(true))
                {
                    try { chart.StopNetMq(); } catch { }
                }
            }
            return;
        }

        // --- Exclusive-swap path (CombinedViewEnabled == false), unchanged ---
        DisableAll(FindObjectsByType<GpuMergedPointCloudLoader>(FindObjectsSortMode.None));
        DisableAll(FindObjectsByType<SimPubRgbdSubscriber>(FindObjectsSortMode.None));
        DisableAll(FindObjectsByType<SimPubPointCloudSubscriber>(FindObjectsSortMode.None));
        DisableAll(FindObjectsByType<GpuPointCloudSubscriber>(FindObjectsSortMode.None));
        DisableAll(FindObjectsByType<PointCloudTest>(FindObjectsSortMode.None));  // class in PointCloudLoader.cs
        DisableAll(FindObjectsByType<SimPubClient>(FindObjectsSortMode.None));
        DisableAll(FindObjectsByType<InterventionStatusHud>(FindObjectsSortMode.None));
        // RGB diamond panel mode: the 4 top/left/right/wrist panels spawned by
        // InterventionRgbPanelSpawner are SessionThumbnailPanel instances, each with
        // its own NetMQ poller. Same crash class as the others above if not stopped
        // proactively before LoadScene (OnDisable already calls the correct
        // blocking-poller.Stop()-before-dispose Shutdown()). Grid selector panels
        // never exist in this scene, so this only ever touches the diamond panels here.
        DisableAll(FindObjectsByType<SessionThumbnailPanel>(FindObjectsSortMode.None));
        // Risk bar chart has N individual subscriber sockets — stop them before scene load.
        foreach (var chart in FindObjectsByType<RiskBarChart>(FindObjectsSortMode.None))
        {
            try { chart.StopNetMq(); } catch { }
        }
    }

    private static void DisableAllInScene<T>(Scene scene) where T : Behaviour
    {
        foreach (var rootGo in scene.GetRootGameObjects())
        {
            foreach (var c in rootGo.GetComponentsInChildren<T>(true))
            {
                if (c != null && c.enabled)
                {
                    try { c.enabled = false; } catch { }
                }
            }
        }
    }

    private static void DisableAll(Behaviour[] components)
    {
        if (components == null) return;
        foreach (var c in components)
        {
            if (c != null && c.enabled)
            {
                try { c.enabled = false; } catch { }
            }
        }
    }
}
