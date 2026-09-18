#if UNITY_EDITOR
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;
using IRIS.SceneLoader;

/// <summary>
/// Editor menu helpers for the VR Multi-Session Grid Selector feature.
///
/// Menu: Tools > Multi-Session > ...
///
/// Run these ONCE after first importing the MultiSession scripts.
/// After running, save both scenes (Ctrl+S) and add them to Build Settings.
/// </summary>
public static class MultiSessionSceneSetup
{
    private const string SelectorScenePath      = "Assets/Scenes/SIIScene_SessionSelector.unity";
    private const string InterventionScenePath  = "Assets/Scenes/SIIScene_InterventionV1.unity";
    private const string McOnlyScenePath        = "Assets/Scenes/SIIScene_MotionControllerOnly.unity";

    // -----------------------------------------------------------------------
    // 1. Set up SIIScene_SessionSelector
    // -----------------------------------------------------------------------

    [MenuItem("Tools/Multi-Session/1. Setup Selector Scene (SIIScene_SessionSelector)")]
    public static void SetupSelectorScene()
    {
        if (!System.IO.File.Exists(SelectorScenePath))
        {
            Debug.LogError($"[MultiSessionSetup] Scene not found: {SelectorScenePath}\n" +
                           "Copy SIIScene_InterventionV1.unity to SIIScene_SessionSelector.unity first.");
            return;
        }

        // Open (or load additively and focus)
        var scene = EditorSceneManager.OpenScene(SelectorScenePath, OpenSceneMode.Single);

        // Remove components that belong to the intervention workflow.
        // Order matters: remove bootstraps/children before parents.

        // --- Wrist camera RGBD panel ---
        RemoveComponentsOfType<SimPubRgbdSubscriber>();
        RemoveComponentsOfType<SimPubRgbdQuad>();   // companion quad renderer

        // --- Point cloud subscribers ---
        RemoveComponentsOfType<SimPubPointCloudSubscriber>();
        RemoveComponentsOfType<GpuPointCloudSubscriber>();
        RemoveComponentsOfType<GpuInstancedPointCloudRenderer>();

        // --- GpuMergedPointCloudBootstrap creates GpuMergedPointCloudLoader at runtime
        //     via AddComponent, so the Loader is absent at edit time.
        //     Removing the Bootstrap prevents the Loader from ever being created. ---
        RemoveComponentsOfType<GpuMergedPointCloudBootstrap>();
        RemoveComponentsOfType<GpuMergedPointCloudLoader>(); // belt-and-suspenders

        // --- SimPubClient (generic ZMQ subscriber — not needed in selector) ---
        RemoveComponentsOfType<SimPubClient>();

        // --- SimSceneSpawner: listens for XR scene-mesh broadcasts and spawns
        //     robot meshes.  Must NOT be active in the selector scene or all 15
        //     MuJoCo sessions would try to spawn overlapping robot meshes. ---
        RemoveComponentsOfType<SimSceneSpawner>();

        // --- SIIMetaQuest3Grabbable is 'internal' so we remove it by type-name
        //     rather than by generic type parameter. ---
        RemoveComponentsByName("SIIMetaQuest3Grabbable");

        // --- Leftover multi-session components from a previous run (idempotency) ---
        RemoveComponentsOfType<InterventionSessionBootstrap>();
        RemoveComponentsOfType<InterventionBackButton>();

        // Add GridManager GameObject (if not already present)
        MultiSessionGridManager gridManager = Object.FindAnyObjectByType<MultiSessionGridManager>();
        if (gridManager == null)
        {
            var go = new GameObject("GridManager");
            gridManager = go.AddComponent<MultiSessionGridManager>();
            Debug.Log("[MultiSessionSetup] Added GridManager to selector scene.");
        }
        else
        {
            Debug.Log("[MultiSessionSetup] GridManager already present.");
        }

        // Add EditorTestHelper to the GridManager GameObject (editor-only, zero device cost)
        if (gridManager.GetComponent<EditorTestHelper>() == null)
        {
            gridManager.gameObject.AddComponent<EditorTestHelper>();
            Debug.Log("[MultiSessionSetup] Added EditorTestHelper to GridManager.");
        }

        // Add NetMQQuitHandler to prevent editor freeze on stop
        EnsureNetMQQuitHandler();

        EditorSceneManager.MarkSceneDirty(scene);
        EditorSceneManager.SaveScene(scene);
        Debug.Log("[MultiSessionSetup] Selector scene saved. Add it to File > Build Settings.");
    }

    // -----------------------------------------------------------------------
    // 2. Set up SIIScene_InterventionV1 (additive only)
    // -----------------------------------------------------------------------

    [MenuItem("Tools/Multi-Session/2. Add Bootstrap + Back Button to SIIScene_InterventionV1")]
    public static void SetupInterventionScene()
    {
        var scene = EditorSceneManager.OpenScene(InterventionScenePath, OpenSceneMode.Single);

        // --- SessionBootstrap ---
        if (Object.FindAnyObjectByType<InterventionSessionBootstrap>() == null)
        {
            var go = new GameObject("SessionBootstrap");
            go.AddComponent<InterventionSessionBootstrap>();
            Debug.Log("[MultiSessionSetup] Added SessionBootstrap to InterventionV1.");
        }
        else
        {
            Debug.Log("[MultiSessionSetup] SessionBootstrap already present.");
        }

        // --- BackButton ---
        if (Object.FindAnyObjectByType<InterventionBackButton>() == null)
        {
            var go = new GameObject("BackButton");
            // Try to parent under CenterEyeAnchor for head-relative positioning
            var anchor = GameObject.Find("CenterEyeAnchor");
            if (anchor != null)
                go.transform.SetParent(anchor.transform, false);
            go.AddComponent<InterventionBackButton>();
            Debug.Log("[MultiSessionSetup] Added BackButton to InterventionV1.");
        }
        else
        {
            Debug.Log("[MultiSessionSetup] BackButton already present.");
        }

        // Add NetMQQuitHandler to prevent editor freeze on stop
        EnsureNetMQQuitHandler();

        EditorSceneManager.MarkSceneDirty(scene);
        EditorSceneManager.SaveScene(scene);
        Debug.Log("[MultiSessionSetup] InterventionV1 scene saved.");
    }

    // -----------------------------------------------------------------------
    // 3. Add both scenes to Build Settings
    // -----------------------------------------------------------------------

    [MenuItem("Tools/Multi-Session/3. Add Scenes to Build Settings")]
    public static void AddScenesToBuildSettings()
    {
        var existing = new System.Collections.Generic.List<EditorBuildSettingsScene>(
            EditorBuildSettings.scenes
        );

        bool addedSelector    = EnsureSceneInBuild(existing, SelectorScenePath);
        bool addedIntervention = EnsureSceneInBuild(existing, InterventionScenePath);

        EditorBuildSettings.scenes = existing.ToArray();

        if (addedSelector || addedIntervention)
        {
            Debug.Log("[MultiSessionSetup] Build settings updated:");
            foreach (var s in existing)
                Debug.Log($"  [{(s.enabled ? "x" : " ")}] {s.path}");
        }
        else
        {
            Debug.Log("[MultiSessionSetup] Both scenes already in Build Settings.");
        }
    }

    // -----------------------------------------------------------------------
    // Motion Controller — Create MC-Only scene
    // -----------------------------------------------------------------------

    [MenuItem("Tools/Motion Controller/Create MC-Only Scene (SIIScene_MotionControllerOnly)")]
    public static void CreateMotionControllerOnlyScene()
    {
        // Create a new empty scene and save it.
        var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

        // ---- OVRCameraRig ----
        // Find the prefab from the project or use any existing OVRCameraRig as a template.
        // Instantiate as a new GameObject named OVRCameraRig if not available as prefab.
        GameObject ovrRigGo = null;
        var ovrRigPrefab = AssetDatabase.LoadAssetAtPath<GameObject>(
            "Assets/Oculus/VR/Prefabs/OVRCameraRig.prefab");
        if (ovrRigPrefab != null)
        {
            ovrRigGo = (GameObject)PrefabUtility.InstantiatePrefab(ovrRigPrefab);
            ovrRigGo.name = "OVRCameraRig";
            Debug.Log("[MCSetup] Instantiated OVRCameraRig prefab.");
        }
        else
        {
            // Fallback: create a minimal camera rig manually.
            ovrRigGo = new GameObject("OVRCameraRig");
            var camGo = new GameObject("CenterEyeAnchor");
            camGo.transform.SetParent(ovrRigGo.transform, false);
            camGo.AddComponent<Camera>();
            Debug.LogWarning("[MCSetup] OVRCameraRig prefab not found — created a minimal fallback. " +
                             "Replace with the real OVRCameraRig prefab for controller tracking.");
        }

        // ---- MC_pub GameObject ----
        // Parented to the OVRCameraRig so it moves with the rig.
        var mcPubGo = new GameObject("MC_pub");
        mcPubGo.transform.SetParent(ovrRigGo.transform, false);
        var mcPub = mcPubGo.AddComponent<MotionControllerZmqPublisher>();
        mcPub.zmqPort   = 6090;
        mcPub.zmqTopic  = "MotionController";
        mcPub.sendHz    = 120;
        // Publisher starts ENABLED in the standalone scene.
        mcPub.enabled   = true;
        Debug.Log("[MCSetup] Added MC_pub GameObject with MotionControllerZmqPublisher.");

        // ---- Status label ----
        var labelGo = new GameObject("MCStatusLabel");
        labelGo.transform.SetParent(ovrRigGo.transform, false);
        var label = labelGo.AddComponent<MotionControllerStatusLabel>();
        label.publisher = mcPub;
        Debug.Log("[MCSetup] Added MCStatusLabel.");

        // ---- NetMQQuitHandler ----
        var quitGo = new GameObject("NetMQQuitHandler");
        quitGo.AddComponent<NetMQQuitHandler>();

        // ---- Save the scene ----
        System.IO.Directory.CreateDirectory("Assets/Scenes");
        bool saved = EditorSceneManager.SaveScene(scene, McOnlyScenePath);
        if (saved)
        {
            Debug.Log($"[MCSetup] Scene saved: {McOnlyScenePath}");

            // Add to Build Settings (as build index 2, after Selector=0, InterventionV1=1).
            var existing = new System.Collections.Generic.List<EditorBuildSettingsScene>(
                EditorBuildSettings.scenes);
            EnsureSceneInBuild(existing, McOnlyScenePath);
            EditorBuildSettings.scenes = existing.ToArray();
            Debug.Log("[MCSetup] Added to Build Settings. "
                    + "IMPORTANT: Do NOT move it to index 0 — Selector must remain at index 0.");
        }
        else
        {
            Debug.LogError($"[MCSetup] Failed to save scene to {McOnlyScenePath}");
        }
    }

    // -----------------------------------------------------------------------
    // Helpers

    private static bool EnsureSceneInBuild(
        System.Collections.Generic.List<EditorBuildSettingsScene> list,
        string path)
    {
        foreach (var s in list)
            if (s.path == path) return false;
        list.Add(new EditorBuildSettingsScene(path, true));
        Debug.Log($"[MultiSessionSetup] Added to Build Settings: {path}");
        return true;
    }

    private static void RemoveComponentsOfType<T>() where T : Component
    {
        foreach (var c in Object.FindObjectsByType<T>(FindObjectsSortMode.None))
        {
            Debug.Log($"[MultiSessionSetup] Removing {typeof(T).Name} from '{c.gameObject.name}'");
            Object.DestroyImmediate(c);
        }
    }

    /// <summary>Adds NetMQQuitHandler to the open scene if not already present.</summary>
    private static void EnsureNetMQQuitHandler()
    {
        if (Object.FindAnyObjectByType<NetMQQuitHandler>() == null)
        {
            var go = new GameObject("NetMQQuitHandler");
            go.AddComponent<NetMQQuitHandler>();
            Debug.Log("[MultiSessionSetup] Added NetMQQuitHandler (prevents editor freeze on stop).");
        }
    }

    /// <summary>
    /// Removes all components whose C# class name matches <paramref name="typeName"/>.
    /// Used for internal/inaccessible types that can't be used as generic type params.
    /// </summary>
    private static void RemoveComponentsByName(string typeName)
    {
        // FindObjectsByType<Component> returns every component in the scene.
        foreach (var c in Object.FindObjectsByType<Component>(FindObjectsSortMode.None))
        {
            if (c == null) continue;
            if (c.GetType().Name == typeName)
            {
                Debug.Log($"[MultiSessionSetup] Removing {typeName} from '{c.gameObject.name}'");
                Object.DestroyImmediate(c);
            }
        }
    }
}
#endif
