using UnityEngine;
using UnityEngine.Rendering;

public enum PointCloudCompositeMode
{
    PerCameraObjects = 0,
    SingleMergedDraw = 1,
}

[DefaultExecutionOrder(-500)]
public class GpuMergedPointCloudBootstrap : MonoBehaviour
{
    private const string BootstrapBuildStamp = "2026-03-17-gpu-merged-draw";

    public bool enableBootstrap = true;
    public bool inheritEndpointFromSimPubClient = true;
    public string publisherIp = "192.168.0.208";
    public int topicPort = 7741;
    public string simSceneName = "MujocoScene";
    public PointCloudCompositeMode compositeMode = PointCloudCompositeMode.SingleMergedDraw;
    public int maxPoints = 100000;
    public int maxPointsPerSource = 100000;
    public int maxCombinedPoints = 300000;
    public float positionScale = 1.0f;
    public float cubeSize = 0.003f;
    public float scale = 1.0f;
    public Material sharedInstancedMaterial;
    public Color fallbackBaseColor = new Color32(255, 255, 255, 242);
    public string pointCloudShaderName = "Custom/StableInstancedPoint";
    public string mergedObjectName = "PointCloudTest_combined";
    public string topObjectName = "PointCloudTest_top";
    public string topTopic = "SimPub/Sensors/top/pc";
    public string rightObjectName = "PointCloudTest_right";
    public string rightTopic = "SimPub/Sensors/right/pc";
    public string leftObjectName = "PointCloudTest_left";
    public string leftTopic = "SimPub/Sensors/left/pc";
    public bool enableDiagnosticProbeMode = false;
    public int diagnosticProbePointCount = 24;
    public float diagnosticProbeCubeSize = 0.02f;
    public PointCloudRenderMode renderMode = PointCloudRenderMode.DepthTestedOpaque;
    public bool useDepthTestDebugMode = false;
    public float debugCubeSizeOverride = 0.0f;
    public bool disableLegacyPointCloudObjects = true;
    public string legacyTopObjectName = "toppc";
    public string legacyRightObjectName = "rightpc";
    public string legacyLeftObjectName = "leftpc";
    public bool logBootstrap = true;

    private Material _runtimeMaterial;
    private bool _loggedMissingSharedMaterialError;
    private bool _loggedFallbackMaterialWarning;
    private bool _loggedFallbackShaderError;
    private bool _loggedBootstrapSummary;
    [SerializeField, HideInInspector] private bool _hasExplicitRenderModeSelection;

    private void Awake()
    {
        if (!enableBootstrap)
            return;

        LogBootstrapSummary();
        ResolveEndpointDefaults();
        if (disableLegacyPointCloudObjects)
            DisableLegacyObjects();

        if (compositeMode == PointCloudCompositeMode.SingleMergedDraw)
        {
            DisableCloudObject(topObjectName);
            DisableCloudObject(rightObjectName);
            DisableCloudObject(leftObjectName);
            EnsureMergedCloudObject();
            return;
        }

        DisableCloudObject(mergedObjectName);
        EnsureCloudObject(topObjectName, topTopic);
        EnsureCloudObject(rightObjectName, rightTopic);
        EnsureCloudObject(leftObjectName, leftTopic);
    }

    private void OnDestroy()
    {
        if (_runtimeMaterial != null)
        {
            if (Application.isPlaying)
                Destroy(_runtimeMaterial);
            else
                DestroyImmediate(_runtimeMaterial);
            _runtimeMaterial = null;
        }
    }

    private void ResolveEndpointDefaults()
    {
        if (!inheritEndpointFromSimPubClient)
            return;

        if (!TryGetComponent(out SimPubClient simPubClient) || simPubClient == null)
            return;

        publisherIp = simPubClient.simpubIp;
        topicPort = simPubClient.simpubTopicPort;
    }

    private void LogBootstrapSummary()
    {
        if (!logBootstrap || _loggedBootstrapSummary)
            return;

        _loggedBootstrapSummary = true;
        Debug.Log(
            "[GpuMergedPointCloudBootstrap] "
            + $"build='{BootstrapBuildStamp}' "
            + $"compositeMode={compositeMode} "
            + $"maxPoints={maxPoints} "
            + $"maxPointsPerSource={maxPointsPerSource} "
            + $"maxCombinedPoints={maxCombinedPoints} "
            + $"renderMode={GetEffectiveRenderMode()} "
            + $"sharedMaterial='{(sharedInstancedMaterial != null ? sharedInstancedMaterial.name : "<none>")}' "
            + "compatibility='PointCloud tag is not required; GPU_Merged lookup uses object names and scene anchors.'"
        );
    }

    private void DisableLegacyObjects()
    {
        DisableLegacyObject(legacyTopObjectName);
        DisableLegacyObject(legacyRightObjectName);
        DisableLegacyObject(legacyLeftObjectName);
    }

    private void DisableLegacyObject(string objectName)
    {
        if (string.IsNullOrWhiteSpace(objectName))
            return;

        var legacy = GameObject.Find(objectName);
        if (legacy == null)
            return;

        if (legacy.activeSelf)
        {
            legacy.SetActive(false);
            if (logBootstrap)
                Debug.Log($"[GpuMergedPointCloudBootstrap] Disabled legacy point-cloud object '{objectName}'.");
        }
    }

    private void DisableCloudObject(string objectName)
    {
        if (string.IsNullOrWhiteSpace(objectName))
            return;

        var cloud = GameObject.Find(objectName);
        if (cloud == null)
            return;

        if (cloud.activeSelf)
        {
            cloud.SetActive(false);
            if (logBootstrap)
                Debug.Log($"[GpuMergedPointCloudBootstrap] Disabled inactive-mode point-cloud object '{objectName}'.");
        }
    }

    private void EnsureCloudObject(string objectName, string topic)
    {
        if (string.IsNullOrWhiteSpace(objectName) || string.IsNullOrWhiteSpace(topic))
            return;

        var go = GameObject.Find(objectName);
        if (go == null)
            go = new GameObject(objectName);
        else if (!go.activeSelf)
            go.SetActive(true);

        go.layer = gameObject.layer;
        go.transform.SetParent(null, false);
        go.transform.localPosition = Vector3.zero;
        go.transform.localRotation = Quaternion.identity;
        go.transform.localScale = Vector3.one;

        var meshFilter = go.GetComponent<MeshFilter>();
        if (meshFilter == null)
            meshFilter = go.AddComponent<MeshFilter>();
        meshFilter.sharedMesh = null;

        var meshRenderer = go.GetComponent<MeshRenderer>();
        if (meshRenderer == null)
            meshRenderer = go.AddComponent<MeshRenderer>();
        var resolvedMaterial = ResolveSharedMaterial();
        meshRenderer.sharedMaterial = resolvedMaterial;
        meshRenderer.enabled = false;
        meshRenderer.shadowCastingMode = ShadowCastingMode.Off;
        meshRenderer.receiveShadows = false;

        var pointCloud = go.GetComponent<PointCloudTest>();
        if (pointCloud == null)
            pointCloud = go.AddComponent<PointCloudTest>();
        pointCloud.enabled = true;

        var mergedPointCloud = go.GetComponent<GpuMergedPointCloudLoader>();
        if (mergedPointCloud != null)
            mergedPointCloud.enabled = false;

        pointCloud.publisherIp = publisherIp;
        pointCloud.topicPort = topicPort;
        pointCloud.topic = topic;
        pointCloud.maxPoints = maxPoints;
        pointCloud.positionScale = positionScale;
        pointCloud.instancedMaterial = resolvedMaterial;
        pointCloud.CubeSize = cubeSize;
        pointCloud.scale = scale;
        pointCloud.fallbackBaseColor = fallbackBaseColor;
        pointCloud.parentToSimScene = true;
        pointCloud.simSceneName = simSceneName;
        pointCloud.sceneAnchorNameOverride = BuildAnchorName(objectName);
        pointCloud.enableDiagnosticProbeMode = enableDiagnosticProbeMode;
        pointCloud.diagnosticProbePointCount = diagnosticProbePointCount;
        pointCloud.diagnosticProbeCubeSize = diagnosticProbeCubeSize;
        pointCloud.SetRenderMode(GetEffectiveRenderMode());
        pointCloud.useDepthTestDebugMode = useDepthTestDebugMode;
        pointCloud.debugCubeSizeOverride = debugCubeSizeOverride;
        pointCloud.logAttachInfo = true;
        pointCloud.ResetBootstrapBindingState();

        if (logBootstrap)
        {
            Debug.Log(
                $"[GpuMergedPointCloudBootstrap] Ready '{objectName}' topic='{topic}' endpoint={publisherIp}:{topicPort} maxPoints={maxPoints} renderMode={GetEffectiveRenderMode()} material='{(resolvedMaterial != null ? resolvedMaterial.name : "<none>")}'."
                + $" build='{BootstrapBuildStamp}'"
            );
        }
    }

    private void EnsureMergedCloudObject()
    {
        var topics = BuildMergedTopics();
        if (topics.Length <= 0 || string.IsNullOrWhiteSpace(mergedObjectName))
            return;

        var go = GameObject.Find(mergedObjectName);
        if (go == null)
            go = new GameObject(mergedObjectName);
        else if (!go.activeSelf)
            go.SetActive(true);

        go.layer = gameObject.layer;
        go.transform.SetParent(null, false);
        go.transform.localPosition = Vector3.zero;
        go.transform.localRotation = Quaternion.identity;
        go.transform.localScale = Vector3.one;

        var meshFilter = go.GetComponent<MeshFilter>();
        if (meshFilter == null)
            meshFilter = go.AddComponent<MeshFilter>();
        meshFilter.sharedMesh = null;

        var meshRenderer = go.GetComponent<MeshRenderer>();
        if (meshRenderer == null)
            meshRenderer = go.AddComponent<MeshRenderer>();
        var resolvedMaterial = ResolveSharedMaterial();
        meshRenderer.sharedMaterial = resolvedMaterial;
        meshRenderer.enabled = false;
        meshRenderer.shadowCastingMode = ShadowCastingMode.Off;
        meshRenderer.receiveShadows = false;

        var pointCloud = go.GetComponent<PointCloudTest>();
        if (pointCloud != null)
            pointCloud.enabled = false;

        var mergedLoader = go.GetComponent<GpuMergedPointCloudLoader>();
        if (mergedLoader == null)
            mergedLoader = go.AddComponent<GpuMergedPointCloudLoader>();
        mergedLoader.enabled = true;

        mergedLoader.publisherIp = publisherIp;
        mergedLoader.topicPort = topicPort;
        mergedLoader.topics = topics;
        mergedLoader.maxPointsPerSource = maxPointsPerSource;
        mergedLoader.maxCombinedPoints = maxCombinedPoints;
        mergedLoader.positionScale = positionScale;
        mergedLoader.instancedMaterial = resolvedMaterial;
        mergedLoader.CubeSize = cubeSize;
        mergedLoader.scale = scale;
        mergedLoader.fallbackBaseColor = fallbackBaseColor;
        mergedLoader.parentToSimScene = true;
        mergedLoader.simSceneName = simSceneName;
        mergedLoader.sceneAnchorNameOverride = BuildAnchorName(mergedObjectName);
        mergedLoader.SetRenderMode(GetEffectiveRenderMode());
        mergedLoader.useDepthTestDebugMode = useDepthTestDebugMode;
        mergedLoader.debugCubeSizeOverride = debugCubeSizeOverride;
        mergedLoader.logAttachInfo = true;
        mergedLoader.ResetBootstrapBindingState();

        if (logBootstrap)
        {
            Debug.Log(
                $"[GpuMergedPointCloudBootstrap] Ready merged '{mergedObjectName}' topics=[{string.Join(", ", topics)}] endpoint={publisherIp}:{topicPort} "
                + $"maxPointsPerSource={maxPointsPerSource} maxCombinedPoints={maxCombinedPoints} renderMode={GetEffectiveRenderMode()} "
                + $"material='{(resolvedMaterial != null ? resolvedMaterial.name : "<none>")}' build='{BootstrapBuildStamp}'."
            );
        }
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

    private Material ResolveSharedMaterial()
    {
        if (sharedInstancedMaterial != null)
        {
            if (sharedInstancedMaterial.shader == null)
            {
                if (!_loggedFallbackShaderError)
                {
                    _loggedFallbackShaderError = true;
                    Debug.LogError("[GpuMergedPointCloudBootstrap] sharedInstancedMaterial is assigned but has no shader. GPU_Merged clouds will remain blocked until a valid material is assigned.");
                }
                return null;
            }

            sharedInstancedMaterial.enableInstancing = true;
            return sharedInstancedMaterial;
        }

        if (!_loggedMissingSharedMaterialError)
        {
            _loggedMissingSharedMaterialError = true;
            Debug.LogError("[GpuMergedPointCloudBootstrap] sharedInstancedMaterial is not assigned. Quest builds should use a serialized material asset; attempting shader fallback.");
        }

        if (_runtimeMaterial != null)
            return _runtimeMaterial;

        if (string.IsNullOrWhiteSpace(pointCloudShaderName))
            return null;

        Shader shader = Shader.Find(pointCloudShaderName);
        if (shader == null)
        {
            if (!_loggedFallbackShaderError)
            {
                _loggedFallbackShaderError = true;
                Debug.LogError(
                    $"[GpuMergedPointCloudBootstrap] Could not find fallback shader '{pointCloudShaderName}' for GPU_Merged clouds."
                );
            }
            return null;
        }

        if (!_loggedFallbackMaterialWarning)
        {
            _loggedFallbackMaterialWarning = true;
            Debug.LogWarning(
                $"[GpuMergedPointCloudBootstrap] Using Shader.Find fallback for '{pointCloudShaderName}'. Assign sharedInstancedMaterial in the scene to make Quest builds deterministic."
            );
        }

        _runtimeMaterial = new Material(shader)
        {
            name = "CubeRenderingMaterial"
        };
        _runtimeMaterial.enableInstancing = true;
        _runtimeMaterial.SetColor("_BaseColor", fallbackBaseColor);
        return _runtimeMaterial;
    }

    private string[] BuildMergedTopics()
    {
        var topics = new System.Collections.Generic.List<string>(3);
        AppendUniqueTopic(topics, topTopic);
        AppendUniqueTopic(topics, rightTopic);
        AppendUniqueTopic(topics, leftTopic);
        return topics.ToArray();
    }

    private static void AppendUniqueTopic(System.Collections.Generic.List<string> topics, string topic)
    {
        if (string.IsNullOrWhiteSpace(topic) || topics.Contains(topic))
            return;

        topics.Add(topic);
    }

    private static string BuildAnchorName(string objectName)
    {
        return $"PointCloudAnchor_{objectName}";
    }
}
